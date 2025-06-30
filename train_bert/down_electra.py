from transformers import ElectraTokenizer, ElectraForSequenceClassification


model_name = "monologg/koelectra-small-finetuned-nsmc"

tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraForSequenceClassification.from_pretrained(model_name)  

#이 코드는 동일한 model_name을 기반으로, 텍스트 분류(task)를 위한 구조로 사전 학습된 electra 모델을 불러옴
#ElectraForSequenceClassificationn은 electra를 기반으로 한 분류(classification) 모델임

save_directory = "./local_electra"

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"모델과 토크나이저가 {save_directory} 에 저장되었습니다.")
