import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from keras.preprocessing.sequence import pad_sequences

from omegaconf import OmegaConf
from argparse import ArgumentParser

def convert_input_data(tokenizer, sentences, cfg):

    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=cfg.max_len, dtype="long", truncating="post", padding="post")

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks

# 문장 테스트
def test_sentences(model, device, tokenizer, sentences, cfg):

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(tokenizer, sentences, cfg)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)

    # 그래디언트 계산 안함
    with torch.no_grad():
        # Forward 수행
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=cfg.num_labels
        )
    model.to(device)
    model.load_state_dict(torch.load(cfg.load_model, map_location=device))

    while True:
      ipt = input()
      if ipt == 'quit':
        break
      logits = test_sentences(model, device, tokenizer, [ipt], cfg)
      print(logits)

      if np.argmax(logits) == 1 :
        print("가스라이팅")
      elif np.argmax(logits) == 0 :
        print("일상 대화")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)