from __future__ import annotations

import argparse
import pickle
import textwrap
import types
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import torch

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None


DEFAULT_GRAPH = Path(__file__).resolve().parent / "step3_graph" / "graph_0_0_0.pt"
TOP_FEATURES = 6
TOP_OUTPUTS = 3
BG_WHITE = "#f3f5f7"
FG_BLACK = "#1c2430"
MUTED_TEXT = "#5e6775"
TITLE_ACCENT = "#2f3e52"
PROMPT_BOX_FILL = "#f8fafc"
PROMPT_BOX_EDGE = "#9aa8bd"
TOKEN_PANEL_FILL = "#eef2f6"
TOKEN_PANEL_EDGE = "#8d99ab"
TOKEN_CHIP_FILL = "#ffffff"
TOKEN_CHIP_EDGE = "#c0c8d4"
FEATURE_FILL = "#f2a65a"
FLOW_COLOR = "#7a879a"
OUTPUT_BOX_FILL = "#fff4e6"
OUTPUT_BOX_EDGE = "#c87f2a"
FOCUS_BOX_FILL = "#eef3fa"
SUMMARY_BOX_FILL = "#f7efe2"


class _FallbackPickleUnpickler(pickle.Unpickler):
    """Allows loading graph payloads without the transformer_lens package."""

    def find_class(self, module: str, name: str):
        if module.startswith("transformer_lens"):
            class _Dummy:  # noqa: N801
                def __setstate__(self, state):
                    if isinstance(state, dict):
                        self.__dict__.update(state)

            _Dummy.__name__ = name
            return _Dummy
        return super().find_class(module, name)


def _fallback_pickle_module() -> types.SimpleNamespace:
    module = types.ModuleType("pickle_fallback")
    module.Unpickler = _FallbackPickleUnpickler
    module.load = pickle.load
    module.loads = pickle.loads
    module.dump = pickle.dump
    module.dumps = pickle.dumps
    return module


def load_graph_payload(path: Path) -> dict:
    try:
        graph = torch.load(path, weights_only=False)
    except ModuleNotFoundError as err:
        if "transformer_lens" not in str(err):
            raise
        graph = torch.load(path, weights_only=False, pickle_module=_fallback_pickle_module())

    if isinstance(graph, dict):
        cfg = graph.get("cfg")
        n_layers = int(getattr(cfg, "n_layers", 1))
        return {
            "input_string": graph["input_string"],
            "input_tokens": graph["input_tokens"],
            "logit_tokens": graph["logit_tokens"],
            "logit_probabilities": graph["logit_probabilities"],
            "active_features": graph["active_features"],
            "activation_values": graph["activation_values"],
            "adjacency_matrix": graph["adjacency_matrix"],
            "selected_features": graph.get("selected_features"),
            "n_layers": n_layers,
        }

    cfg = getattr(graph, "cfg", None)
    return {
        "input_string": graph.input_string,
        "input_tokens": graph.input_tokens,
        "logit_tokens": graph.logit_tokens,
        "logit_probabilities": graph.logit_probabilities,
        "active_features": graph.active_features,
        "activation_values": graph.activation_values,
        "adjacency_matrix": graph.adjacency_matrix,
        "selected_features": getattr(graph, "selected_features", None),
        "n_layers": int(getattr(cfg, "n_layers", 1)),
    }


def load_tokenizer() -> GPT2TokenizerFast | None:
    if GPT2TokenizerFast is None:
        return None
    try:
        return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        return None


def format_token(tokenizer: GPT2TokenizerFast | None, token_id: int) -> str:
    if tokenizer is None:
        return f"token {token_id}"
    decoded = tokenizer.decode([int(token_id)])
    decoded = decoded.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    return decoded or "space"


def wrap_lines(text: str, width: int) -> str:
    return textwrap.fill(" ".join(str(text).split()), width=width)


def top_feature_records(payload: dict) -> list[dict]:
    active_features = payload["active_features"]
    activation_values = payload["activation_values"]
    top_indices = torch.argsort(activation_values, descending=True)[:TOP_FEATURES].tolist()

    records = []
    for rank, index in enumerate(top_indices, start=1):
        layer, pos, feature_idx = active_features[index].tolist()
        records.append(
            {
                "active_index": int(index),
                "graph_index": int(index),
                "rank": rank,
                "layer": int(layer),
                "pos": int(pos),
                "feature_idx": int(feature_idx),
                "activation": float(activation_values[index]),
            }
        )
    return records


def top_output_records(payload: dict, tokenizer: GPT2TokenizerFast | None) -> list[dict]:
    order = torch.argsort(payload["logit_probabilities"], descending=True)[:TOP_OUTPUTS].tolist()
    return [
        {
            "logit_index": int(index),
            "token": format_token(tokenizer, int(payload["logit_tokens"][index])),
            "probability": float(payload["logit_probabilities"][index]),
        }
        for index in order
    ]


def important_token_records(payload: dict, tokenizer: GPT2TokenizerFast | None, features: list[dict]) -> list[dict]:
    seen_positions: list[int] = []
    for feature in features:
        if feature["pos"] not in seen_positions:
            seen_positions.append(feature["pos"])

    records = []
    input_tokens = payload["input_tokens"].tolist()
    for pos in seen_positions[:TOP_FEATURES]:
        if pos >= len(input_tokens):
            continue
        records.append({"pos": pos, "token": format_token(tokenizer, int(input_tokens[pos]))})
    return records


def feature_to_output_edges(payload: dict, features: list[dict], outputs: list[dict]) -> list[dict]:
    adjacency = payload["adjacency_matrix"]
    selected_features = payload.get("selected_features")
    if selected_features is None:
        return []

    selected_list = [int(x) for x in selected_features.tolist()]
    selected_lookup = {active_idx: col_idx for col_idx, active_idx in enumerate(selected_list)}
    n_features = len(selected_list)
    n_tokens = len(payload["input_tokens"])
    token_start = n_features + payload["n_layers"] * n_tokens
    logit_start = token_start + n_tokens
    logit_block = adjacency[logit_start : logit_start + len(payload["logit_tokens"]), :n_features]

    edges = []
    for output in outputs:
        for feature in features:
            selected_col = selected_lookup.get(feature["active_index"])
            if selected_col is None:
                continue
            weight = float(logit_block[output["logit_index"], selected_col])
            if abs(weight) > 0:
                edges.append(
                    {
                        "feature_rank": feature["rank"],
                        "logit_index": output["logit_index"],
                        "weight": weight,
                    }
                )
    edges.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return edges[:6]


def draw_rounded_box(ax, x: float, y: float, w: float, h: float, face: str, edge: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012",
            linewidth=1.6,
            facecolor=face,
            edgecolor=edge,
        )
    )


def draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    lw: float,
    style: str = "solid",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=lw,
            color=color,
            linestyle=style,
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.86,
        )
    )


def draw_storyboard(payload: dict, output_path: Path) -> None:
    tokenizer = load_tokenizer()
    features = top_feature_records(payload)
    outputs = top_output_records(payload, tokenizer)
    important_tokens = important_token_records(payload, tokenizer, features)
    output_edges = feature_to_output_edges(payload, features, outputs)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=180)
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "CodeCircuit Step3 Graph: Prompt To Attribution To Predicted Tokens",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=FG_BLACK,
    )
    ax.plot([0.02, 0.98], [0.92, 0.92], color=TITLE_ACCENT, lw=1.4)

    ax.text(0.16, 0.84, "Prompt Context", ha="center", va="center", fontsize=15, fontweight="bold", color=TITLE_ACCENT)
    ax.text(0.50, 0.84, "Attribution Graph", ha="center", va="center", fontsize=15, fontweight="bold", color=TITLE_ACCENT)
    ax.text(0.84, 0.84, "Model Output", ha="center", va="center", fontsize=15, fontweight="bold", color=TITLE_ACCENT)

    draw_rounded_box(ax, 0.04, 0.48, 0.24, 0.27, PROMPT_BOX_FILL, PROMPT_BOX_EDGE)
    ax.text(0.055, 0.735, wrap_lines(payload["input_string"], 32), ha="left", va="top", fontsize=10.2, color=FG_BLACK)

    draw_rounded_box(ax, 0.04, 0.21, 0.24, 0.19, TOKEN_PANEL_FILL, TOKEN_PANEL_EDGE)
    ax.text(0.055, 0.375, "Top influential prompt tokens", ha="left", va="center", fontsize=11, fontweight="bold", color=TITLE_ACCENT)
    for idx, token_info in enumerate(important_tokens):
        y = 0.335 - idx * 0.04
        label = f"token {token_info['pos']}: {token_info['token']}"
        ax.text(0.06, y, label, ha="left", va="center", fontsize=10, color=FG_BLACK)

    draw_arrow(ax, (0.30, 0.52), (0.36, 0.52), color=FLOW_COLOR, lw=2.0)

    center_left, center_right = 0.37, 0.67
    token_y, output_y = 0.12, 0.73

    unique_layers = sorted({f["layer"] for f in features})
    layer_positions = {}
    if len(unique_layers) <= 1:
        layer_positions[unique_layers[0] if unique_layers else 0] = 0.46
    else:
        y_values = torch.linspace(0.28, 0.63, len(unique_layers)).tolist()
        for layer, y_val in zip(unique_layers, y_values):
            layer_positions[layer] = float(y_val)

    max_pos = max([record["pos"] for record in important_tokens] + [1])
    token_positions = {
        record["pos"]: center_left + (center_right - center_left) * (record["pos"] / max_pos)
        for record in important_tokens
    }

    feature_positions = {}
    for feature in features:
        x = token_positions.get(feature["pos"], center_left + 0.02)
        if feature["rank"] % 2 == 0:
            x += 0.012
        feature_positions[feature["rank"]] = (x, layer_positions.get(feature["layer"], 0.46))

    output_positions = {}
    for idx, output in enumerate(outputs):
        step = (center_right - center_left) / max(1, len(outputs) - 1)
        output_positions[output["logit_index"]] = (center_left + idx * step, output_y)

    for layer in unique_layers:
        y = layer_positions[layer]
        ax.text(0.345, y, f"Layer {layer}", ha="right", va="center", fontsize=10.5, color=MUTED_TEXT)
        ax.plot([center_left - 0.01, center_right + 0.01], [y, y], color="#d8d3c6", lw=0.8, linestyle="--", alpha=0.7)

    for record in important_tokens:
        x = token_positions[record["pos"]]
        draw_rounded_box(ax, x - 0.032, token_y - 0.02, 0.064, 0.045, TOKEN_CHIP_FILL, TOKEN_CHIP_EDGE)
        ax.text(x, token_y + 0.002, record["token"], ha="center", va="center", fontsize=10.4, color=FG_BLACK)

    for output in outputs:
        x, y = output_positions[output["logit_index"]]
        draw_rounded_box(ax, x - 0.04, y - 0.022, 0.08, 0.05, TOKEN_CHIP_FILL, TOKEN_CHIP_EDGE)
        ax.text(x, y + 0.002, output["token"], ha="center", va="center", fontsize=10.3, color=FG_BLACK)

    for feature in features:
        x, y = feature_positions[feature["rank"]]
        radius = 0.010 + 0.008 * max(0.0, feature["activation"])
        ax.add_patch(Circle((x, y), radius=radius, facecolor=FEATURE_FILL, edgecolor="#ffffff", linewidth=1.1, alpha=0.95))
        ax.text(x, y - 0.035, f"F{feature['feature_idx']}", ha="center", va="center", fontsize=8.2, color=MUTED_TEXT)

    for feature in features:
        fx, fy = feature_positions[feature["rank"]]
        tx = token_positions.get(feature["pos"])
        if tx is not None:
            draw_arrow(ax, (tx, token_y + 0.028), (fx, fy - 0.018), color=FLOW_COLOR, lw=1.9, style="dashed", rad=0.08)

    for edge in output_edges:
        fx, fy = feature_positions[edge["feature_rank"]]
        ox, oy = output_positions[edge["logit_index"]]
        lw = 1.5 + 3.0 * min(abs(edge["weight"]), 0.35)
        draw_arrow(ax, (fx, fy + 0.02), (ox, oy - 0.03), color=FLOW_COLOR, lw=lw, rad=0.16)

    draw_arrow(ax, (0.69, 0.52), (0.74, 0.52), color=FLOW_COLOR, lw=2.0)

    if outputs:
        draw_rounded_box(ax, 0.76, 0.68, 0.19, 0.09, OUTPUT_BOX_FILL, OUTPUT_BOX_EDGE)
        ax.text(0.855, 0.725, f"Top next token: {outputs[0]['token']}", ha="center", va="center", fontsize=11.4, color=FG_BLACK)
        ax.text(0.855, 0.695, f"Probability: {outputs[0]['probability'] * 100:.1f}%", ha="center", va="center", fontsize=10, color=MUTED_TEXT)

    active_layers = ", ".join(str(layer) for layer in unique_layers) if unique_layers else "n/a"
    strongest = f"F{features[0]['feature_idx']}" if features else "n/a"
    draw_rounded_box(ax, 0.76, 0.50, 0.19, 0.10, FOCUS_BOX_FILL, OUTPUT_BOX_EDGE)
    ax.text(0.855, 0.555, "Where the model focused", ha="center", va="center", fontsize=11.4, color=FG_BLACK)
    ax.text(0.855, 0.525, f"Layers: {active_layers}", ha="center", va="center", fontsize=10, color=MUTED_TEXT)
    ax.text(0.855, 0.497, f"Strongest feature: {strongest}", ha="center", va="center", fontsize=10, color=MUTED_TEXT)

    summary = (
        "Prompt tokens activate hidden features; those feature activations"
        " then drive the model's top next-token preferences."
    )
    draw_rounded_box(ax, 0.76, 0.30, 0.19, 0.13, SUMMARY_BOX_FILL, OUTPUT_BOX_EDGE)
    ax.text(0.855, 0.39, "What this means", ha="center", va="center", fontsize=11.4, color=FG_BLACK)
    ax.text(0.855, 0.345, wrap_lines(summary, 26), ha="center", va="center", fontsize=9.5, color=MUTED_TEXT)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Step3 attribution graph storyboard PNG.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Path to a step3 .pt graph file")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph.resolve()
    output_path = args.output.resolve() if args.output else graph_path.with_name(f"{graph_path.stem}_storyboard.png")
    payload = load_graph_payload(graph_path)
    draw_storyboard(payload, output_path)
    print(f"Saved step3 storyboard image to {output_path}")


if __name__ == "__main__":
    main()
