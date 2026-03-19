import json
from pathlib import Path

input_file = Path("task1/step1_line_correctness.jsonl")
output_file = Path("task1/step2_incremental.jsonl")


def load_responses(path: Path):
    responses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses

def generate_incremental_records(responses):
    records = []
    for item in responses:
        task_id = item.get("task_id")
        task_text = item.get("text", "")
        code = item.get("generated_code", "")
        scores = item.get("line_correctness_scores", [])
        lines = code.split("\n")
        for i in range(1, len(lines)+1):
            # label = correctness of the last line in the current cumulative snippet
            label = scores[i-1] if i <= len(scores) else 1
            new_record = {
                "task_id": task_id,
                "text": task_text + " " + "\n".join(lines[:i]),
                "generated_code": "\n".join(lines[:i]),
                "label": label
            }
            records.append(new_record)
    return records

def save_records(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if not input_file.exists():
        raise FileNotFoundError(f"Input not found: {input_file}. Run Step 1 first.")

    responses = load_responses(input_file)
    incremental_records = generate_incremental_records(responses)
    save_records(incremental_records, output_file)
    print(f"Generated {len(incremental_records)} incremental records and saved to {output_file}")