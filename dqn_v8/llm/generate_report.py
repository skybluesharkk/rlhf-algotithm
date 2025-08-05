from urllib.parse import urlparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from transformers import StoppingCriteria, StoppingCriteriaList
import re
class StopAtNewline(StoppingCriteria):
    def __init__(self, tok): self.id = tok.encode("\n")[0]
    def __call__(self, ids, scores, **kwargs): return ids[0, -1] == self.id

MODEL_NAME = "google/gemma-7b-it"

def build_prompt(i: int, j: int) -> str:
    actions = [
        "[-1,0,0]", "[1,0,0]", "[0,1,0]", "[0,0,0.8]",
        "[-1,1,0]", "[1,1,0]", "[0,0,0]"
    ]
    return (
        "Task: Calculate car action similarity with these STRICT rules:\n"
        "\n"
        "Action meanings:\n"
        "- steer: -1=left, +1=right, 0=straight\n"
        "- throttle: acceleration (higher = faster)\n"
        "- brake: deceleration (higher = stronger braking)\n"
        "\n"
        f"a={actions[i]}\n"
        f"b={actions[j]}\n"
        "\n"
        "Check rules first, then calculate. Output only decimal number:\n"
        "score: "
    )

def main():
    print(f"모델 로드 중: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map={"": 0}, torch_dtype=torch.float16
    )

    nlp = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False         
    )
    stopper = StoppingCriteriaList([StopAtNewline(tokenizer)])
    for i in range(7):              
        for j in range(7):
            prompt = build_prompt(i, j)
            out = nlp(prompt, max_new_tokens=3, do_sample=False,
                      eos_token_id=tokenizer.eos_token_id)[0]["generated_text"]
            num = re.search(r"[-+]?\d*\.?\d+", out)
            print(f"({i},{j}) -> {num.group() if num else out.strip()}")
main()