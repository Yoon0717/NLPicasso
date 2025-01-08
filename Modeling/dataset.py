import re
import numpy as np
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    def __init__(self, tokenizer, chats, max_len, Q_TKN, A_TKN, SENT, EOS, MASK):
        self.tokenizer = tokenizer
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.mask = MASK

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["prompt"]
        q = re.sub(r"([.!,])", r" ", q)

        a = turn["competition"]
        a = re.sub(r"([.!,])", r" ", a)

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        labels = [self.mask,] * q_len + a_toked[1:]
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)

        label_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(label_ids) < self.max_len:
            label_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(mask), label_ids)