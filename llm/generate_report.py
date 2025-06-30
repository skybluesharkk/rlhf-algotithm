#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from urllib.parse import urlparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# ───────────────────────────────────────────────────────────────
# 1) 사용할 공개 모델 지정
#    - Microsoft Phi-2 (2.7B 파라미터, 공개 모델)
MODEL_NAME = "microsoft/phi-2"
# ───────────────────────────────────────────────────────────────

# 2) mock_data 하드코딩 (URL, input_fields, query_params)
mock_data = [
    (
        "http://suninatas.com/",
        [ {"type": "text",     "name": "id"},
          {"type": "password", "name": "pw"} ],
        {}
    ),
    (
        "http://suninatas.com/redirect?url=https://www.facebook.com/suninatas",
        [],
        {"url": "https://www.facebook.com/suninatas"}
    ),
    (
        "http://suninatas.com/redirect?url=http://www.teruten.com/kr/index.php",
        [],
        {"url": "http://www.teruten.com/kr/index.php"}
    ),
    (
        "http://suninatas.com/chatting",
        [],
        {}
    ),
    (
        "http://suninatas.com/board/notice",
        [ {"name": "searchT"},
          {"type": "text", "name": "searchK", "value": ""} ],
        {}
    ),
]

def get_crawling_summary():
    total_links       = len(mock_data)
    unique_hosts      = len({ urlparse(u).netloc for u,_,_ in mock_data })
    pages_with_inputs = sum(1 for _, inp, _ in mock_data if inp)
    pages_with_params = sum(1 for _, _, prm in mock_data if prm)
    return {
        "total_links": total_links,
        "unique_hosts": unique_hosts,
        "pages_with_inputs": pages_with_inputs,
        "pages_with_params": pages_with_params,
        "sample_data": mock_data
    }

def build_prompt(summary: dict) -> str:
    parts = [
        "다음은 크롤링된 웹 페이지의 통계 및 샘플 데이터입니다.",
        "아래 정보를 기반으로 주요 공격 벡터를 식별하고 설명하시오.\n",
        "[크롤링 요약]",
        f"- 전체 링크 수: {summary['total_links']}",
        f"- 고유 호스트 수: {summary['unique_hosts']}",
        f"- 입력 필드가 있는 페이지 수: {summary['pages_with_inputs']}",
        f"- 쿼리 파라미터가 있는 페이지 수: {summary['pages_with_params']}\n",
        "[샘플 데이터]"
    ]
    for url, inputs, params in summary["sample_data"]:
        parts.append(f"URL: {url}")
        parts.append(f"  - input_fields: {json.dumps(inputs, ensure_ascii=False)}")
        parts.append(f"  - query_params: {json.dumps(params, ensure_ascii=False)}\n")
    return "\n".join(parts)

def main():
    # 1) 요약 생성
    summary = get_crawling_summary()
    prompt  = build_prompt(summary)

    # 2) 모델·토크나이저 로드 (MPS/CPU 자동 분배, FP16 가속)
    print(f"▶ 모델 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 3) 생성 파이프라인 구성
    nlp = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # 4) 보안 보고서 생성
    print("▶ 보안 보고서 생성 중…")
    result = nlp(
        prompt,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.2,
        num_beams=4,
        no_repeat_ngram_size=3
    )
    report_text = result[0]["generated_text"]

    # 5) 결과 저장
    out_path = "security_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✔ 보안 보고서를 '{out_path}'에 저장했습니다.")

if __name__ == "__main__":
    main()
