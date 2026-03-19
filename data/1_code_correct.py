import ast
import json
from pathlib import Path

from tqdm import tqdm

# Step 1 for your custom Step 0 output.
INPUT_FILE = Path("task1/gemma_mbpp_responses.jsonl")
OUTPUT_FILE = Path("task1/step1_line_correctness.jsonl")


def load_responses(path: Path):
    responses = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def score_lines_python(code: str) -> list[int]:
    """Return per-line 1/0 correctness scores.

    Rule used here:
    - If full code parses: all lines are 1.
    - If syntax error at line N: lines < N are 1, lines >= N are 0.
    """
    lines = code.splitlines() or [code]
    n_lines = max(1, len(lines))

    try:
        ast.parse(code)
        return [1] * n_lines
    except SyntaxError as err:
        err_line = err.lineno or 1
        err_line = max(1, min(err_line, n_lines))
        return [1 if i < err_line else 0 for i in range(1, n_lines + 1)]
    except Exception:
        return [0] * n_lines


def process(responses: list[dict]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as w:
        for item in tqdm(responses, desc="Step1 scoring"):
            code = item.get("generated_code", "")
            scores = score_lines_python(code)
            item["line_correctness_scores"] = scores
            w.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}. Run Step 0 first to create it."
        )

    responses = load_responses(INPUT_FILE)
    process(responses)
    print(f"Saved {len(responses)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
