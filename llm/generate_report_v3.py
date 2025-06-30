#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from urllib.parse import urlparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ───────────────────────────────────────────────────────────────
# 1) Instruction-tuned public model to use
#    - Example on Hugging Face: mosaicml/mpt-7b-instruct
MODEL_NAME = "mosaicml/mpt-7b-instruct"
# ───────────────────────────────────────────────────────────────

# 2) Hardcoded mock_data (URL, input_fields, query_params)
mock_data = [
    # Login page with username/password fields
    (
        "http://suninatas.com/",
        [{"type": "text", "name": "id"}, {"type": "password", "name": "pw"}],
        {}
    ),
    # Open redirect endpoints
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
    # Chatting endpoint (no inputs)
    (
        "http://suninatas.com/chatting",
        [],
        {}
    ),
    # Notice board with search field
    (
        "http://suninatas.com/board/notice",
        [{"type": "text", "name": "searchK", "value": ""}],
        {}
    ),
    # SQL injection test payload
    (
        "http://suninatas.com/login",
        [{"type": "text", "name": "id", "value": "' OR '1'='1"},
         {"type": "password", "name": "pw", "value": "password"}],
        {}
    ),
    # XSS test payload in search
    (
        "http://suninatas.com/board/notice",
        [{"type": "text", "name": "searchK", "value": "<script>alert('XSS')</script>"}],
        {}
    ),
    # Brute force vulnerable login (rate limiting missing)
    (
        "http://suninatas.com/login",
        [{"type": "text", "name": "id", "value": "admin"},
         {"type": "password", "name": "pw", "value": "123456"}],
        {}
    ),
    # Additional endpoint with file upload scenario
    (
        "http://suninatas.com/upload",
        [{"type": "file", "name": "avatar"}],
        {}
    ),
]


def get_crawling_summary():
    total_links = len(mock_data)
    unique_hosts = len({urlparse(u).netloc for u, _, _ in mock_data})
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
    warning = (
        "WARNING: You must strictly follow the response format below without any deviations, typos, or extra text.\n"
        "Use the exact URLs provided; do not alter or misspell domains.\n"
        "Respond only with the numbered analysis result (1 to 4) and no additional commentary.\n"
    )
    examples = (
        "Example format (for reference only):\n"
        "1. SQL Injection:\n"
        "   - Description: This vulnerability allows attackers to manipulate database queries by injecting SQL code into input fields.\n"
        "   - Affected URL: http://suninatas.com/ (id, pw fields)\n"
        "   - Example payload: ' OR '1'='1\n"
        "   - Mitigation: Use input validation and parameterized queries.\n\n"
        "Use the above format as a reference. Do NOT copy this example. Your analysis must be based on the actual sample data below.\n"
    )

    parts = [
        warning,
        "Below is the summary and sample data of crawled web pages. Based on this information, identify and explain at least four (4) major attack vectors using the numbered format shown in the examples above.",
        examples,
        "[Crawling Summary]",
        f"- Total links crawled: {summary['total_links']}",
        f"- Unique hosts: {summary['unique_hosts']}",
        f"- Pages with input fields: {summary['pages_with_inputs']}",
        f"- Pages with query parameters: {summary['pages_with_params']}",
        "",
        "[Sample Data]"
    ]
    for url, inputs, params in summary["sample_data"]:
        parts.append(f"URL: {url}")
        parts.append(f"  - input_fields: {json.dumps(inputs, ensure_ascii=False)}")
        parts.append(f"  - query_params: {json.dumps(params, ensure_ascii=False)}")
        parts.append("")
    parts.append("### Analysis Result:")
    return "\n".join(parts)


def validate_response(response: str) -> bool:
    import re
    required_patterns = [
        r"1\.\s*SQL Injection",
        r"2\.\s*Cross[- ]Site Scripting",
        r"3\.\s*Open Redirect",
        r"4\.\s*(Authentication Bypass|Brute Force)"
    ]
    return all(re.search(p, response, re.IGNORECASE) for p in required_patterns)


def main():
    summary = get_crawling_summary()
    prompt = build_prompt(summary)

    print(f"▶ Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )

    nlp = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False
    )

    print("▶ Generating security report…")
    result = nlp(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.9,
        num_beams=1,
        no_repeat_ngram_size=3
    )
    report_text = result[0]["generated_text"]

    out_path = "security_report7.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"✔ Security report saved to '{out_path}'")


if __name__ == "__main__":
    main()
