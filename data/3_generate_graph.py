from pathlib import Path
import torch
import json
from circuit_tracer import ReplacementModel, attribute
from tqdm import tqdm

# --- Configuration and Model Loading ---
MODEL_NAME = "Qwen/Qwen3-0.6B"
TRANSCODER_NAME = "mwhanna/qwen3-0.6b-transcoders-lowl0"

# --- Graph Generation Parameters ---
MAX_N_LOGITS = 10
DESIRED_LOGIT_PROB = 0.95
MAX_FEATURE_NODES = 8192
BATCH_SIZE = 64
OFFLOAD = None
VERBOSE = False

# --- Input/Output Paths ---
DATA_FILE_PATH = Path("task1/step2_incremental.jsonl")
OUTPUT_DIR = Path("task1/step3_graph")

OUTPUT_DIR.mkdir(exist_ok=True)
print(f"Graph files will be saved in: {OUTPUT_DIR.resolve()}")

# Output metadata file (jsonl append mode)
METADATA_FILE_PATH = OUTPUT_DIR / 'graph_metadata.jsonl'


def load_data(file_path: Path) -> list:
    data = []
    with file_path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def append_metadata(metadata: dict):
    """Append one metadata entry to the jsonl file."""
    with open(METADATA_FILE_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata) + "\n")


def generate_and_save_graphs(data_list: list, output_dir: Path, model: ReplacementModel):
    task_step_counters = {}
    error_log_path = output_dir / "step3_errors.jsonl"

    for i, item in enumerate(tqdm(data_list, desc="Generating Graphs")):
        prompt = item['text']
        task_id = item['task_id']
        current_step = task_step_counters.get(task_id, 0)

        graph_name = f'graph_{i}_{task_id}_{current_step}.pt'
        graph_path = output_dir / graph_name

        # --- Skip if graph exists ---
        if graph_path.exists():
            print(f"\nSkipping generation for existing file: {graph_name}")

            metadata = {
                'expr_id': task_id,
                'step_number': current_step,
                'before_after': 'after',
                'graph_path': str(graph_path.resolve()),
                'step_labels': item['label'],
                'original_expression': item['text'],
            }
            append_metadata(metadata)

            task_step_counters[task_id] = current_step + 1
            continue

        # --- Generation ---
        if len(prompt) > 550:
            continue

        try:
            graph = attribute(
                prompt=prompt,
                model=model,
                max_n_logits=MAX_N_LOGITS,
                desired_logit_prob=DESIRED_LOGIT_PROB,
                batch_size=BATCH_SIZE,
                max_feature_nodes=MAX_FEATURE_NODES,
                offload=OFFLOAD,
                verbose=VERBOSE
            )

            graph.to_pt(graph_path)

            metadata = {
                'expr_id': task_id,
                'step_number': current_step,
                'before_after': 'after',
                'graph_path': str(graph_path.resolve()),
                'step_labels': item['label'],
                'original_expression': item['text'],
            }

            append_metadata(metadata)
            task_step_counters[task_id] = current_step + 1
        except Exception as exc:
            error_record = {
                "index": i,
                "task_id": task_id,
                "step_number": current_step,
                "error": str(exc),
            }
            with error_log_path.open("a", encoding="utf-8") as ef:
                ef.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            print(f"Failed graph for index={i}, task_id={task_id}: {exc}")


if __name__ == "__main__":
    if not DATA_FILE_PATH.exists():
        raise FileNotFoundError(f"Input not found: {DATA_FILE_PATH}. Run Step 2 first.")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ReplacementModel.from_pretrained(
        MODEL_NAME,
        TRANSCODER_NAME,
        dtype=dtype,
    ).to(device)

    # Clear metadata file if already exists
    METADATA_FILE_PATH.open('w', encoding='utf-8').close()

    data_list = load_data(DATA_FILE_PATH)
    print(f"Loaded {len(data_list)} incremental records")
    generate_and_save_graphs(data_list, OUTPUT_DIR, model)

    print("\n--- Process Complete ---")
    print(f"Metadata appended to: {METADATA_FILE_PATH.resolve()}")
