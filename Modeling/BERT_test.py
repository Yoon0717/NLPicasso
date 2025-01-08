import torch
import pandas as pd
import numpy as np
import random
import time
import datetime

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from keras.preprocessing.sequence import pad_sequences

from omegaconf import OmegaConf
from argparse import ArgumentParser

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_data(gaslight_data, chatbot_data):
    df_gas = pd.read_csv(gaslight_data)
    df_chatbot = pd.read_csv(chatbot_data)

    df_gas['label'] = 1
    df_chatbot['label'] = 0
    df_gas.columns = ["prompt", "competition",	"label"]
    df_chatbot.columns = ["prompt", "competition",	"label"]

    df_all = pd.concat([df_gas, df_chatbot], axis=0, ignore_index=True)
    df_all.drop(['prompt'], axis=1, inplace=True)

    return df_all

# 정확도 계산 함수
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# 시간 표시 함수
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))

    # hh:mm:ss으로 형태 변경
    return str(datetime.timedelta(seconds=elapsed_rounded))

def test(device, model, test_dataloader):
    # 테스트 시작 시간
    t0 = time.time()
    
    # 모델 평가 모드 설정
    model.eval()
    
    # 초기화
    eval_accuracy = 0
    nb_eval_steps = 0
    
    # 테스트 데이터셋 평가
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 출력
        if step % 100 == 0 and step > 0:
            elapsed = format_time(time.time() - t0)
            print(f"  Batch {step:>5} of {len(test_dataloader):>5}. Elapsed: {elapsed}.")
    
        # 데이터를 디바이스로 이동
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
    
        # 그래디언트 비활성화 (추론 모드)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs.logits
    
        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
    
        # 정확도 계산
        eval_accuracy += flat_accuracy(logits, label_ids)
        nb_eval_steps += 1
    
    # 최종 정확도 출력
    avg_accuracy = eval_accuracy / nb_eval_steps
    print(f"\nAccuracy: {avg_accuracy:.2f}")
    print(f"Test took: {format_time(time.time() - t0)}")


def main(cfg):
    set_seed(cfg.random_seed)
    df_all = get_data(cfg.gaslight_data, cfg.chatbot_data)

    test_data = df_all[13000:]

    # [CLS] + 문장 + [SEP]
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in test_data.competition]

    labels = test_data['label'].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 시퀀스 설정 및 정수 인덱스 변환 & 패딩
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=cfg.max_len, dtype="long", truncating="post", padding="post")

    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    test_inputs = torch.tensor(input_ids)
    test_labels = torch.tensor(labels)
    test_masks = torch.tensor(attention_masks)

    # 배치 사이즈 설정 및 데이터 설정
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=cfg.num_labels
        )
    model.to(device)
    model.load_state_dict(torch.load(cfg.load_model, map_location=device))

    test(device, model, test_dataloader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)