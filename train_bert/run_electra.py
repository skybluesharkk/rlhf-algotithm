from transformers import ElectraTokenizer, ElectraForSequenceClassification
import torch
import time
# 로컬에 저장된 모델 디렉토리 경로
local_model_path = "./local_electra"
start = time.time()
# 로컬에서 토크나이저와 모델 불러오기
tokenizer = ElectraTokenizer.from_pretrained(local_model_path)
model = ElectraForSequenceClassification.from_pretrained(local_model_path)

# 장치 설정 (GPU 있으면 GPU로)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 테스트 문장을 통한 추론 예제
test_text = "이 영화 정말 재밌어요!"
inputs = tokenizer(
    test_text,
    return_tensors="pt",
    padding="max_length",
    max_length=128,
    truncation=True
)
inputs = {key: val.to(device) for key, val in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

print("예측 결과:", "긍정" if prediction == 1 else "부정")
end = time.time()
mid = end - start
print(start,end,'total:',mid)