import torch
import pandas as pd
import numpy as np
import random
import time
import datetime
import copy

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf
from argparse import ArgumentParser

from optimizer import OptimizerSelector
from Modeling.scheduler import SchedulerSelector

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

def train(device, model, train_dataloader, validation_dataloader, optimizer, scheduler, cfg):
    model.zero_grad()

    # Validation 정확도와 모델 가중치를 저장할 리스트 초기화
    top_models = []

    # Training
    for epoch_i in range(cfg.epoch):
        print(f"\n======== Epoch {epoch_i + 1} / {cfg.epoch} ========")
        print("Training...")

        # 로스 초기화
        total_loss = 0

        # 훈련모드로 변경
        model.train()

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for step, batch in enumerate(train_dataloader):
            # 경과 정보 표시
            if step % 500 == 0 and step > 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step:>5} of {len(train_dataloader):>5}. Elapsed: {elapsed}.")

            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()

            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 옵티마이저와 스케줄러 업데이트
            optimizer.step()
            scheduler.step()

            # 그래디언트 초기화
            model.zero_grad()

        # 평균 로스 계산
        avg_train_loss = total_loss / len(train_dataloader)

        print(f"\n  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")

        # Validation
        print("\nRunning Validation...")

        # 시작 시간 설정
        t0 = time.time()

        # 평가모드로 변경
        model.eval()

        # 변수 초기화
        eval_accuracy = 0
        nb_eval_steps = 0

        # 데이터로더에서 배치만큼 반복하여 가져옴
        for batch in validation_dataloader:
            # 배치를 GPU에 넣음
            batch = tuple(t.to(device) for t in batch)

            # 배치에서 데이터 추출
            b_input_ids, b_input_mask, b_labels = batch

            # 그래디언트 계산 안함
            with torch.no_grad():
                # Forward 수행
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # 로짓 계산
            logits = outputs.logits

            # CPU로 데이터 이동
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # 정확도 계산
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        # 평균 Validation Accuracy 계산
        avg_val_accuracy = eval_accuracy / nb_eval_steps
        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation took: {format_time(time.time() - t0)}")

        # 모델 가중치와 Validation Accuracy 저장
        top_models.append((avg_val_accuracy, copy.deepcopy(model.state_dict())))

        # Validation 결과를 정확도 기준으로 정렬 (내림차순)
        top_models = sorted(top_models, key=lambda x: x[0], reverse=True)

        # 상위 3개 모델만 유지
        if len(top_models) > 3:
            top_models = top_models[:3]

    # 상위 3개 모델 저장
    for i, (accuracy, model_state) in enumerate(top_models):
        model_save_path = f"top_model_{i+1}_accuracy_{accuracy:.2f}.pt"
        torch.save(model_state, model_save_path)
        print(f"Model {i+1} saved with accuracy: {accuracy:.2f} at {model_save_path}")

    print("\nTraining complete!")


def main(cfg):
    set_seed(cfg.random_seed)
    df_all = get_data(cfg.gaslight_data, cfg.chatbot_data)

    train_data = df_all[:13000]

    sentences = ["[CLS] " + str(s) + " [SEP]" for s in train_data.competition]

    labels = train_data['label'].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    tokenized_texts = [tokenizer.tokenize(s) for s in sentences]

    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=cfg.max_len, dtype="long", truncating="post", padding="post")

    # attention mask
    attention_masks = []

    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 훈련셋, 검증셋 분리
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids,
                                                                                        labels,
                                                                                        cfg.random_state,
                                                                                        test_size=0.1)

    train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                           input_ids,
                                                           cfg.random_state,
                                                           test_size=0.1)

    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)
    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    batch_size = cfg.batch_size

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=cfg.num_labels
        )
    model.to(device)

    optimizer_selector = OptimizerSelector('adam', model, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
    optimizer = optimizer_selector.get_optim()

    sched = SchedulerSelector(cfg.scheduler, optimizer, cfg.max_epoch)
    scheduler = sched.get_sched()

    train(device, model, train_dataloader, validation_dataloader, optimizer, scheduler, cfg)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)