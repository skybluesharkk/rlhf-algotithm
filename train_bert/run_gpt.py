import openai
import os
from dotenv import load_dotenv
import time
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") 

start = time.time()
# 분석할 문장
test_text = "이 영화 정말 재밌어요!"

# GPT API에 전달할 프롬프트 구성
prompt = f"다음 문장의 감정을 분석해줘. 답변은 반드시 '긍정' 또는 '부정'으로만 해줘.\n문장: {test_text}"

# gpt-3.5-turbo (Chat 모델) 이용한 감성 분석 호출
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
       {"role": "system", "content": "너는 한국어 감성 분석 전문가야."},
       {"role": "user", "content": prompt}
    ],
    temperature=0,  # 결과의 일관성을 위해 낮은 temperature 값 사용
    max_tokens=10   # 간단한 응답을 위해 충분한 토큰 수 설정
)

# 응답에서 결과 추출
result = response.choices[0].message.content.strip()
print("예측 결과:", result)
end = time.time()
mid = end - start
print(start,end,'total:',mid)