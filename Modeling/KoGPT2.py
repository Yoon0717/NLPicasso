import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer, models, util

from omegaconf import OmegaConf
from argparse import ArgumentParser

from dataset import ChatbotDataset

# Tokens
Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'
UNK = '<unk>'

# Define collate function
def collate_batch(batch, koGPT2_TOKENIZER):
    max_len = 200 # 최대 길이
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]

    for i in range(len(data)):
        if len(data[i]) < max_len:
            padding_length = max_len - len(data[i])
            data[i] = data[i] + [koGPT2_TOKENIZER.pad_token_id] * padding_length
            mask[i] = list(mask[i]) + [0] * padding_length
            label[i] = label[i] + [koGPT2_TOKENIZER.pad_token_id] * padding_length

    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

def main(cfg):
    Chatbot_Data = pd.read_csv(cfg.data_root)
    Chatbot_Data = Chatbot_Data[:3000]
 
    # 토크나이저 로드
    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer, bos_token=BOS, eos_token=EOS, unk_token=UNK, pad_token=PAD, mask_token=MASK)

    max_lengths = []
    max_len=200
    train_set = ChatbotDataset(koGPT2_TOKENIZER, Chatbot_Data, max_len, Q_TKN, A_TKN, SENT, EOS, MASK)
    train_dataloader = DataLoader(train_set, batch_size=10, num_workers=0, shuffle=True, collate_fn=collate_batch)
    
    for i in range(len(train_set)):
        token_ids, _, _ = train_set[i]
        max_lengths.append(len(token_ids))

    print("Maximum length in the dataset:", max(max_lengths))

    # 결과 확인
    print("start")
    for batch_idx, samples in enumerate(train_dataloader):
        token_ids, mask, label = samples
        print("token_ids ====> ", token_ids)
        print("mask =====> ", mask)
        print("label =====> ", label)
    print("end")

    # 모델 사용
    model = torch.load(cfg.model_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Chat with the trained model
    with torch.no_grad():
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            a = ""
            while 1:
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)
                model = model.to('cpu')
                pred = model(input_ids)
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            print("Chatbot > {}".format(a.strip()))
    
    embedding_model = models.Transformer(
        model_name_or_path=cfg.sts_root,
        max_seq_length=256,
        do_lower_case=True
    )

    pooling_model = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    model = SentenceTransformer(modules=[embedding_model, pooling_model])

    docs = a # gpt 답변
    
    #각 문장의 vector값 encoding
    document_embeddings = model.encode(docs)

    query = cfg.sample_text
    query_embedding = model.encode(query)

    # 코사인 유사도 계산 후,
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    print(f"입력 문장: {query}")
    print(f'gpt 답변: {a}')
    print(round(cos_scores.item(),3))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)