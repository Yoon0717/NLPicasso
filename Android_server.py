import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, PreTrainedTokenizerFast, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, models, util
from sentence_transformers import SentenceTransformer

from flask_ngrok import run_with_ngrok
from flask import Flask, request, jsonify

app = Flask(__name__)
run_with_ngrok(app)  # Flask 앱을 ngrok에 연결

# bert 토크나이저
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
device_bert = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_bert():
    model_bert = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
    model_bert.load_state_dict(torch.load("/content/drive/MyDrive/NLPicasso/최종/bert_finetuned_gas.pth"))  # 미리 학습한 가중치 로드
    model_bert.to(device_bert)
    model_bert.eval()
    return model_bert

# gpt model load
def load_model_gpt():
    model_gpt = torch.load("/content/drive/MyDrive/NLPicasso/최종/fine_tuned_model_1.pt")  # 미리 학습한 가중치 로드

    BOS = "<usr>"
    EOS = "<sys>"
    SENT = '<unused1>'
    PAD = "<pad>"
    MASK = "<unused0>"

    # Load tokenizer
    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token=MASK)

    return model_gpt, koGPT2_TOKENIZER


def predict_bert(model, message):
    # 주어진 메시지에 대해 모델 예측 수행
    inputs = tokenizer_bert(message, return_tensors='pt', padding=True, truncation=True)
    inputs = inputs.to(device_bert)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predictions = np.argmax(logits.cpu().numpy(), axis=1)

    if predictions[0] == 1:
        return "가스라이팅 문장입니다."
    else:
        return "일상 대화 문장입니다."


# GPT 평가
def predict_gpt(my_sentence):
    model, koGPT2_TOKENIZER = load_model_gpt()

    # Training parameters
    learning_rate = 3e-5
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch = 10
    Sneg = -1e18

    Q_TKN = "<usr>"
    A_TKN = "<sys>"
    BOS = "</s>"
    EOS = "</s>"
    SENT = '<unused1>'
    PAD = "<pad>"
    MASK = "<unused0>"

    # Load tokenizer
    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token=MASK)

    responses = []


    # Chat with the trained model
    with torch.no_grad():
        q = my_sentence
        q = my_sentence.strip()
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
        return a

"""
# STS
def predict_sts(message, my_sentence):
    embedding_model = models.Transformer(
        model_name_or_path="KDHyun08/TAACO_STS",
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

    docs = predict_gpt(my_sentence) # gpt 답변
    #각 문장의 vector값 encoding
    document_embeddings = model.encode(docs)

    query = message
    query_embedding = model.encode(query)

    # 코사인 유사도 계산 후,
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # print(f"입력 문장: {query}")
    # print(f'gpt 답변: {a}')
    print(int(cos_scores.item() * 100))
    similarity = int(cos_scores.item() * 100)
    return similarity
"""

# sentence trnasformer
def predict_sts(message, my_sentence):
  sent1 = message
  sent2 = predict_gpt(my_sentence)
  model = SentenceTransformer('paraphrase-distilroberta-base-v1')

  def cosine(sent1, sent2):
    sentences = [sent1,sent2]
    # 임베딩
    sentence_embeddings = model.encode(sentences)
    # 유사도 공식
    return np.dot(sentence_embeddings[0],sentence_embeddings[1])/(np.linalg.norm(sentence_embeddings[0])*np.linalg.norm(sentence_embeddings[1]))

  similarity = int(cosine(sent1,sent2) * 100)

  if similarity < 50:
      similarity += 50
  return similarity


# Model load
model_bert = load_model_bert()
model_gpt = load_model_gpt()


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        message = data.get("message")
        my_sentence = data.get("my_sentence")

        # 메시지 분석
        prediction = predict_bert(model_bert, message)

        if prediction == "가스라이팅 문장입니다.":
            result = {"result": predict_sts(message, my_sentence)}
            return jsonify(result)
        elif prediction == "일상 대화 문장입니다.":
            result = {"result": prediction}  # 분석 결과 반환
            return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()