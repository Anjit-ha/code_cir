from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
import os

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=dtype,
)

data = [{"task_id": 0, "prompt": "write a python function to check if a number is prime or not"}]

# 固定的 system_prompt：强制模型只输出纯代码
SYSTEM_PROMPT = (
    "You are a Python coding assistant. "
    "You must output ONLY valid Python code. "
    "Do not include comments, markdown, explanations, or text. "
    "Output only code."
)

def build_prompt(example):
    user_prompt = example["prompt"]
    return SYSTEM_PROMPT + "\n\n" + user_prompt


def clean_output(text: str):
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    text = re.sub(r'("""|\'\'\')(.*?)\1', '', text, flags=re.DOTALL)
    lines = text.splitlines()
    code_lines = [line for line in lines if not line.strip().startswith("#")]
    code_lines = [line for line in code_lines if line.strip() != ""]
    return "\n".join(code_lines).strip()

def run_example(example, max_new_tokens=48):
    prompt = build_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return clean_output(generated)

results = []
for i, ex in enumerate(data):
    code = run_example(ex)
    print(f"===== Example {i} =====")
    print(code)
    print("\n")
    results.append({
        "task_id": ex.get("task_id", i),
        "text": ex["prompt"],
        "generated_code": code,
    })

output_path = os.path.join("task1", "gemma_mbpp_responses.jsonl")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

