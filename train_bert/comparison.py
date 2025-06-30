import time
import torch
from transformers import ElectraTokenizer, ElectraForSequenceClassification
import openai
import os
from dotenv import load_dotenv

# 환경 변수 로드 및 OpenAI API 키 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 테스트에 사용할 입력 문장
test_text = "앞으로 잘 가고 있어!"

#############################################
# 1. Electra 기반 로컬 모델 추론 측정
#############################################

# 로컬 모델이 저장된 경로 (GPU 서버에 맞게 미리 저장된 모델 사용)
local_model_path = "./local_electra"

# 모델과 토크나이저 로드 (초기 로드 시간은 별도로 측정하지 않도록)
electra_tokenizer = ElectraTokenizer.from_pretrained(local_model_path)
electra_model = ElectraForSequenceClassification.from_pretrained(local_model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
electra_model.to(device)
electra_model.eval()

# 추론 함수를 정의 (여러 번 반복 측정을 위해)
def measure_electra_inference():
    # 입력 전처리
    inputs = electra_tokenizer(
        test_text,
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    # 추론 시간 측정 시작
    start_time = time.time()
    with torch.no_grad():
        outputs = electra_model(**inputs)
    elapsed_time = time.time() - start_time
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    result = "긍정" if prediction == 1 else "부정"
    print(f"Electra 예측 결과: {result}, 실행 시간: {elapsed_time:.4f}초")
    return elapsed_time

# 여러 번 측정하여 평균 시간을 확인할 수 있습니다.
electra_times = [measure_electra_inference() for _ in range(5)]
average_electra = sum(electra_times) / len(electra_times)
print(f"Electra 평균 추론 시간: {average_electra:.4f}초\n")

#############################################
# 2. GPT API 기반 추론 측정
#############################################

def measure_gpt_inference():
    # GPT API에 전달할 프롬프트 구성
    prompt = f"다음 문장의 감정을 분석해줘. 답변은 반드시 '긍정' 또는 '부정'으로만 해줘.\n문장: {test_text}"
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "너는 한국어 감성 분석 전문가야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=10
    )
    elapsed_time = time.time() - start_time
    result = response.choices[0].message.content.strip()
    print(f"GPT 예측 결과: {result}, 실행 시간: {elapsed_time:.4f}초")
    return elapsed_time

# GPT 추론도 여러 번 실행해 평균 시간을 산출해봅니다.
gpt_times = [measure_gpt_inference() for _ in range(5)]
average_gpt = sum(gpt_times) / len(gpt_times)
print(f"GPT 평균 추론 시간: {average_gpt:.4f}초")
