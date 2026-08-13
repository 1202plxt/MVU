#!/usr/bin/env python
"""
CVBench Attention Visualization — 3 correct + 3 incorrect samples

Loads the reasoning_analysis.json from the previous step, re-runs inference
on the selected samples with attention extraction, and generates detailed
visualizations showing WHERE the model attends across videos.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Safe Layer Count Extraction
# ═══════════════════════════════════════════════════════════════════════════
def get_num_layers(model) -> int:
    """Safely extract total hidden layers across different config structures."""
    config = model.config
    
    for attr in ["num_hidden_layers", "num_layers", "n_layer"]:
        if hasattr(config, attr):
            return getattr(config, attr)
            
    for sub_cfg_name in ["text_config", "llm_config", "language_config"]:
        if hasattr(config, sub_cfg_name):
            sub_cfg = getattr(config, sub_cfg_name)
            for attr in ["num_hidden_layers", "num_layers", "n_layer"]:
                if hasattr(sub_cfg, attr):
                    return getattr(sub_cfg, attr)

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "layers"):
        return len(model.layers)

    raise AttributeError("Could not determine layer count from model config or architecture.")


# ═══════════════════════════════════════════════════════════════════════════
# Video frame extraction
# ═══════════════════════════════════════════════════════════════════════════
def extract_frames(video_path: str, num_frames: int = 4) -> List[Image.Image]:
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)
        if total == 0:
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except ImportError:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════
def load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            model_cls = AutoModelForCausalLM

    logging.info(f"Loading model from {model_path}")
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    return model, processor


# ═══════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════
def build_prompt(item: dict, video_frames: Dict[str, List[Image.Image]]):
    content_parts = []
    all_images = []
    video_labels = []

    for vkey in ["video_1", "video_2", "video_3", "video_4"]:
        vpath = item.get(vkey)
        if vpath is None or vpath not in video_frames:
            continue
        frames = video_frames[vpath]
        if not frames:
            continue
        content_parts.append({"type": "text", "text": f"\n[{vkey}]:"})
        for frame in frames:
            content_parts.append({"type": "image", "image": frame})
            all_images.append(frame)
            video_labels.append(vkey)

    question = item["question"]
    options = item.get("options", [])
    opts_text = "\n".join(options) if options else ""
    if opts_text:
        q_text = (
            f"\n\nQuestion: {question}\nOptions:\n{opts_text}\n\n"
            "Please select the correct answer and explain briefly."
        )
    else:
        q_text = f"\n\nQuestion: {question}\nPlease answer briefly."
    content_parts.append({"type": "text", "text": q_text})

    messages = [{"role": "user", "content": content_parts}]
    return messages, all_images, video_labels


# ═══════════════════════════════════════════════════════════════════════════
# Inference with multi-layer attention extraction
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def inference_with_attention(
    model, processor, messages, all_images,
    max_new_tokens: int = 512,
    layers_to_capture: Optional[List[int]] = None,
):
    # =========================================================================
    # Step 1: Forward Pass (Generation)
    # =========================================================================
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    # Map image tokens from prompt
    image_token_id = None
    for name in ["<|image_pad|>", "<|vision_start|>", "<image>"]:
        tid = processor.tokenizer.convert_tokens_to_ids(name)
        if tid != processor.tokenizer.unk_token_id:
            image_token_id = tid
            break
            
    if image_token_id is not None:
        image_token_mask = (input_ids[0] == image_token_id).cpu()
    else:
        image_token_mask = torch.zeros(prompt_len, dtype=torch.bool)

    # Generate
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
    )
    output_ids = generated_ids[0, prompt_len:]
    gen_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)
    gen_len = output_ids.shape[0]
    logging.info(f"  Generated {gen_len} tokens.")

    # =========================================================================
    # Step 2: Extract Attention (via re-processing to fix mRoPE shape issues)
    # =========================================================================
    if not gen_text.strip():
        gen_text = " "
        
    # Append the answer back to messages and re-apply chat template
    # This offloads complex mask/position_id aligning back to the processor!
    messages.append({"role": "assistant", "content": gen_text})
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    forward_inputs = processor(
        text=[full_text], 
        images=all_images if all_images else None, 
        padding=True, 
        return_tensors="pt"
    )
    forward_inputs = {k: v.to(model.device) for k, v in forward_inputs.items()}
    forward_inputs["output_attentions"] = True
    forward_inputs["use_cache"] = False
    
    actual_full_len = forward_inputs["input_ids"].shape[1]
    actual_gen_len = actual_full_len - prompt_len

    num_layers = get_num_layers(model)
    if layers_to_capture is None:
        layers_to_capture = sorted(set([
            0,
            num_layers // 4,
            num_layers // 2,
            3 * num_layers // 4,
            num_layers - 1,
        ]))
    logging.info(f"  Extracting attention from layers: {layers_to_capture}")

    per_layer_attn = {}
    try:
        outputs = model(**forward_inputs)
        
        for li in layers_to_capture:
            if li < len(outputs.attentions):
                attn = outputs.attentions[li]  # shape: (1, heads, full_len, full_len)
                # Slice: generated answer tokens attending back to prompt tokens
                sliced = attn[0, :, prompt_len:, :prompt_len]
                per_layer_attn[li] = sliced.cpu().float()

        del outputs
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        logging.warning("  OOM during attention extraction!")
        torch.cuda.empty_cache()

    return gen_text, {
        "per_layer_attn": per_layer_attn,
        "image_token_mask": image_token_mask,  # Perfect align with prompt_len
        "input_len": prompt_len,
        "gen_len": gen_len,
        "generated_ids": output_ids.cpu(),
        "generated_text": gen_text,
        "num_layers": num_layers,
        "layers_captured": layers_to_capture,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Attention analysis helpers
# ═══════════════════════════════════════════════════════════════════════════
def compute_per_video_attention(
    attn: torch.Tensor,               # (heads, actual_gen_len, prompt_len)
    image_token_mask: torch.Tensor,   # (prompt_len,)
    video_labels: List[str],
    input_len: int,
) -> Tuple[Dict[str, float], torch.Tensor]:
    avg_attn = attn.mean(dim=0)       # (actual_gen_len, prompt_len)
    
    img_positions = image_token_mask.nonzero(as_tuple=True)[0]
    n_img = len(img_positions)
    if n_img == 0:
        return {}, torch.zeros(avg_attn.shape[0], 1)

    attn_to_img = avg_attn[:, img_positions]  # (actual_gen_len, n_img)
    n_frames = len(video_labels)
    tokens_per_frame = max(n_img // n_frames, 1)

    per_frame = torch.zeros(avg_attn.shape[0], n_frames)
    for fi in range(n_frames):
        s = fi * tokens_per_frame
        e = min((fi + 1) * tokens_per_frame, n_img)
        if e > s:
            per_frame[:, fi] = attn_to_img[:, s:e].sum(dim=1)

    unique_videos = list(dict.fromkeys(video_labels))
    video_attn = {}
    for vname in unique_videos:
        frame_indices = [i for i, vl in enumerate(video_labels) if vl == vname]
        video_attn[vname] = per_frame[:, frame_indices].sum(dim=1).mean().item()

    total = sum(video_attn.values())
    if total > 0:
        video_attn = {k: v / total for k, v in video_attn.items()}

    return video_attn, per_frame


def compute_cross_video_ratio_by_layer(
    attn_data: dict,
    video_labels: List[str],
) -> Dict[int, float]:
    ratios = {}
    for li, attn in attn_data["per_layer_attn"].items():
        video_fracs, _ = compute_per_video_attention(
            attn,
            attn_data["image_token_mask"],
            video_labels,
            attn_data["input_len"],
        )
        if not video_fracs:
            ratios[li] = 0.0
            continue
        vals = list(video_fracs.values())
        n = len(vals)
        if n <= 1:
            ratios[li] = 1.0
        else:
            mean_v = sum(vals) / n
            std_v = (sum((v - mean_v) ** 2 for v in vals) / n) ** 0.5
            max_std = ((n - 1) / n) ** 0.5
            ratios[li] = 1.0 - (std_v / max_std) if max_std > 0 else 1.0
    return ratios


# ═══════════════════════════════════════════════════════════════════════════
# Color palette
# ═══════════════════════════════════════════════════════════════════════════
VIDEO_COLORS = {
    "video_1": "#3B82F6",
    "video_2": "#EF4444",
    "video_3": "#22C55E",
    "video_4": "#A855F7",
}
CORRECT_COLOR = "#22C55E"
INCORRECT_COLOR = "#EF4444"
BG_COLOR = "#1A1A2E"
CARD_COLOR = "#16213E"
TEXT_COLOR = "#E0E0E0"
SUBTLE_TEXT = "#8899AA"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "attn_heat", ["#1A1A2E", "#0F3460", "#E94560", "#FFD460"], N=256
)


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample visualization
# ═══════════════════════════════════════════════════════════════════════════
def visualize_single_sample(
    item: dict,
    video_frames: Dict[str, List[Image.Image]],
    video_labels: List[str],
    attn_data: dict,
    processor,
    save_path: str,
    is_correct: bool,
):
    last_layer = max(attn_data["per_layer_attn"].keys())
    last_attn = attn_data["per_layer_attn"][last_layer]

    video_fracs, per_frame_attn = compute_per_video_attention(
        last_attn, attn_data["image_token_mask"], video_labels, attn_data["input_len"]
    )
    layer_evenness = compute_cross_video_ratio_by_layer(attn_data, video_labels)

    gen_ids = attn_data["generated_ids"]
    gen_text = attn_data["generated_text"]
    gen_labels = []
    for tid in gen_ids:
        tok = processor.tokenizer.decode([tid.item()])
        gen_labels.append(tok.strip() if tok.strip() else "·")

    unique_videos = list(dict.fromkeys(video_labels))
    frame_images = []
    for vname in unique_videos:
        for vk in ["video_1", "video_2", "video_3", "video_4"]:
            if vk == vname:
                vpath = item.get(vk)
                if vpath and vpath in video_frames:
                    frame_images.extend(video_frames[vpath])
                break

    n_frames = len(video_labels)
    max_gen_show = min(40, len(gen_labels))

    fig = plt.figure(figsize=(22, 20), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(
        4, 1, height_ratios=[1.5, 2, 4, 2],
        hspace=0.30, left=0.06, right=0.94, top=0.90, bottom=0.04,
    )

    status = "CORRECT" if is_correct else "INCORRECT"
    status_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR

    q_short = item["question"][:120] + ("…" if len(item["question"]) > 120 else "")
    pred = item.get("prediction", "?")
    gt = item.get("ground_truth", item.get("answer", "?"))

    fig.suptitle(
        f"Sample #{item.get('id', '?')}  [{item.get('task_type', '')}]  —  {status}",
        fontsize=16, fontweight="bold", color=status_color,
        y=0.96,
    )
    fig.text(
        0.50, 0.93,
        f"Q: {q_short}",
        ha="center", fontsize=10, color=TEXT_COLOR,
        fontstyle="italic",
    )
    fig.text(
        0.50, 0.91,
        f"Pred: {pred}    GT: {gt}    Generated: {gen_text[:80]}{'…' if len(gen_text) > 80 else ''}",
        ha="center", fontsize=9, color=SUBTLE_TEXT,
    )

    # Row 0: Frame thumbnails
    gs_frames = gridspec.GridSpecFromSubplotSpec(
        1, n_frames, subplot_spec=gs[0], wspace=0.05
    )
    for fi in range(n_frames):
        ax = fig.add_subplot(gs_frames[fi], facecolor=CARD_COLOR)
        if fi < len(frame_images):
            ax.imshow(frame_images[fi])
        ax.set_xticks([])
        ax.set_yticks([])
        vl = video_labels[fi]
        c = VIDEO_COLORS.get(vl, "#888888")
        ax.set_title(f"{vl} · f{fi}", fontsize=8, color=c, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color(c)
            spine.set_linewidth(2.5)

    # Row 1: Per-video attention share
    ax1 = fig.add_subplot(gs[1], facecolor=CARD_COLOR)
    if video_fracs:
        vnames = list(video_fracs.keys())
        vvals = [video_fracs[v] for v in vnames]
        vcolors = [VIDEO_COLORS.get(v, "#888") for v in vnames]
        bars = ax1.bar(vnames, vvals, color=vcolors, width=0.5, edgecolor="white",
                       linewidth=0.5, alpha=0.9)
        for bar, val in zip(bars, vvals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_COLOR,
            )
    ax1.set_ylabel("Attention Fraction", fontsize=10, color=TEXT_COLOR)
    ax1.set_title(
        "Per-Video Attention Share (last layer, avg over generated tokens)",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10,
    )
    ax1.set_ylim(0, 1.0)
    ax1.set_facecolor(CARD_COLOR)
    ax1.tick_params(colors=TEXT_COLOR)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax1.spines[spine].set_color(SUBTLE_TEXT)

    # Row 2: Per-frame attention heatmap
    ax2 = fig.add_subplot(gs[2], facecolor=CARD_COLOR)
    pf = per_frame_attn[:max_gen_show, :n_frames]
    row_sums = pf.sum(dim=1, keepdim=True).clamp(min=1e-8)
    pf_norm = (pf / row_sums).numpy().T

    im = ax2.imshow(pf_norm, aspect="auto", cmap=HEATMAP_CMAP, interpolation="nearest")
    ax2.set_yticks(range(n_frames))
    ylabels = [f"{video_labels[i]}:f{i}" for i in range(n_frames)]
    ax2.set_yticklabels(ylabels, fontsize=8, color=TEXT_COLOR)

    for tl in ax2.get_yticklabels():
        vname = tl.get_text().split(":")[0]
        tl.set_color(VIDEO_COLORS.get(vname, TEXT_COLOR))

    ax2.set_xticks(range(max_gen_show))
    ax2.set_xticklabels(gen_labels[:max_gen_show], fontsize=7, rotation=60,
                        ha="right", color=TEXT_COLOR)
    ax2.set_xlabel("Generated Tokens →", fontsize=10, color=TEXT_COLOR)
    ax2.set_ylabel("Video Frames", fontsize=10, color=TEXT_COLOR)
    ax2.set_title(
        "Per-Frame Attention Heatmap (which frames does each output token attend to?)",
        fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10,
    )
    cbar = plt.colorbar(im, ax=ax2, shrink=0.6, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=9, color=TEXT_COLOR)
    cbar.ax.tick_params(colors=TEXT_COLOR)

    # Row 3: Cross-video evenness across layers
    ax3 = fig.add_subplot(gs[3], facecolor=CARD_COLOR)
    if layer_evenness:
        layers_sorted = sorted(layer_evenness.keys())
        evenness_vals = [layer_evenness[l] for l in layers_sorted]
        ax3.plot(layers_sorted, evenness_vals, "o-", color="#E94560",
                 linewidth=2, markersize=6, markerfacecolor="#FFD460",
                 markeredgecolor="white", markeredgewidth=1)
        ax3.fill_between(layers_sorted, evenness_vals, alpha=0.15, color="#E94560")
        ax3.set_xlabel("Layer Index", fontsize=10, color=TEXT_COLOR)
        ax3.set_ylabel("Cross-Video Evenness", fontsize=10, color=TEXT_COLOR)
        ax3.set_title(
            "Attention Evenness Across Layers (1.0 = equally attending all videos, 0.0 = one video dominates)",
            fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=10,
        )
        ax3.set_ylim(-0.05, 1.05)
        ax3.axhline(y=1.0 / len(unique_videos), color=SUBTLE_TEXT, linestyle="--",
                    linewidth=1, alpha=0.5, label="Random baseline")
        ax3.legend(fontsize=8, facecolor=CARD_COLOR, edgecolor=SUBTLE_TEXT,
                    labelcolor=TEXT_COLOR)
    ax3.tick_params(colors=TEXT_COLOR)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax3.spines[spine].set_color(SUBTLE_TEXT)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Comparative visualization
# ═══════════════════════════════════════════════════════════════════════════
def visualize_comparison(
    correct_data: List[dict],
    incorrect_data: List[dict],
    save_path: str,
):
    n_correct = len(correct_data)
    n_incorrect = len(incorrect_data)
    n_rows = max(n_correct, n_incorrect)

    fig = plt.figure(figsize=(20, 6 * n_rows + 6), facecolor=BG_COLOR)
    gs_main = gridspec.GridSpec(
        n_rows + 1, 2,
        hspace=0.40, wspace=0.25,
        left=0.07, right=0.93, top=0.93, bottom=0.04,
    )

    fig.suptitle(
        "Attention Distribution: Correct vs Incorrect Samples",
        fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.97,
    )

    def draw_sample_bar(ax, sample_info, is_correct):
        video_fracs = sample_info["video_fracs"]
        sid = sample_info["id"]
        task = sample_info["task_type"]
        pred = sample_info.get("pred", "?")
        gt = sample_info.get("gt", "?")

        status_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR
        ax.set_facecolor(CARD_COLOR)

        if video_fracs:
            vnames = list(video_fracs.keys())
            vvals = [video_fracs[v] for v in vnames]
            vcolors = [VIDEO_COLORS.get(v, "#888") for v in vnames]
            bars = ax.barh(vnames, vvals, color=vcolors, height=0.5,
                           edgecolor="white", linewidth=0.5, alpha=0.9)
            for bar, val in zip(bars, vvals):
                ax.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%",
                    ha="left", va="center", fontsize=10, color=TEXT_COLOR,
                )

        ax.set_xlim(0, 1.0)
        title = f"#{sid} [{task}]  Pred={pred}  GT={gt}"
        ax.set_title(title, fontsize=10, fontweight="bold", color=status_color, pad=8)
        ax.tick_params(colors=TEXT_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color(SUBTLE_TEXT)

    for i in range(n_rows):
        if i < n_correct:
            ax_l = fig.add_subplot(gs_main[i, 0])
            draw_sample_bar(ax_l, correct_data[i], is_correct=True)
            if i == 0:
                ax_l.text(
                    0.5, 1.35, "CORRECT SAMPLES",
                    transform=ax_l.transAxes, ha="center", fontsize=13,
                    fontweight="bold", color=CORRECT_COLOR,
                )
        if i < n_incorrect:
            ax_r = fig.add_subplot(gs_main[i, 1])
            draw_sample_bar(ax_r, incorrect_data[i], is_correct=False)
            if i == 0:
                ax_r.text(
                    0.5, 1.35, "INCORRECT SAMPLES",
                    transform=ax_r.transAxes, ha="center", fontsize=13,
                    fontweight="bold", color=INCORRECT_COLOR,
                )

    ax_bot = fig.add_subplot(gs_main[n_rows, :], facecolor=CARD_COLOR)

    for i, sd in enumerate(correct_data):
        le = sd.get("layer_evenness", {})
        if le:
            layers = sorted(le.keys())
            vals = [le[l] for l in layers]
            ax_bot.plot(layers, vals, "o-", color=CORRECT_COLOR, alpha=0.6,
                        linewidth=1.5, markersize=4,
                        label=f"Correct #{sd['id']}" if i == 0 else None)

    for i, sd in enumerate(incorrect_data):
        le = sd.get("layer_evenness", {})
        if le:
            layers = sorted(le.keys())
            vals = [le[l] for l in layers]
            ax_bot.plot(layers, vals, "s--", color=INCORRECT_COLOR, alpha=0.6,
                        linewidth=1.5, markersize=4,
                        label=f"Incorrect #{sd['id']}" if i == 0 else None)

    ax_bot.set_xlabel("Layer Index", fontsize=11, color=TEXT_COLOR)
    ax_bot.set_ylabel("Attention Evenness", fontsize=11, color=TEXT_COLOR)
    ax_bot.set_title(
        "Cross-Video Attention Evenness by Layer (correct vs incorrect)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=10,
    )
    ax_bot.set_ylim(-0.05, 1.05)
    ax_bot.legend(fontsize=9, facecolor=CARD_COLOR, edgecolor=SUBTLE_TEXT,
                  labelcolor=TEXT_COLOR, loc="lower right")
    ax_bot.tick_params(colors=TEXT_COLOR)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    for s in ["bottom", "left"]:
        ax_bot.spines[s].set_color(SUBTLE_TEXT)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logging.info(f"Comparison figure saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CVBench Attention Visualization")
    parser.add_argument("--model_path", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/shared_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/datasets/CVBench")
    parser.add_argument("--analysis_json", type=str, default="",
                        help="Path to reasoning_analysis.json from previous step")
    parser.add_argument("--output_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/outputs/cvbench_attention_vis")
    parser.add_argument("--num_frames", type=int, default=4)
    
    # 调整了最大 token 数量，防止截断推理
    parser.add_argument("--max_new_tokens", type=int, default=512) 
    
    parser.add_argument("--correct_ids", type=int, nargs="*", default=None,
                        help="Manually specify correct sample IDs")
    parser.add_argument("--incorrect_ids", type=int, nargs="*", default=None,
                        help="Manually specify incorrect sample IDs")
    parser.add_argument("--num_layers", type=int, default=5,
                        help="Number of layers to sample for attention (evenly spaced)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = data_dir / "CVBench.json"
    logging.info(f"Loading CVBench from {json_path}")
    with open(json_path) as f:
        all_data = json.load(f)
    id_to_item = {item.get("id", None): item for item in all_data}

    target_correct_ids = args.correct_ids
    target_incorrect_ids = args.incorrect_ids

    if target_correct_ids is None or target_incorrect_ids is None:
        if not args.analysis_json:
            default_analysis = (
                Path(args.data_dir).parent.parent
                / "outputs" / "cvbench_reasoning" / "reasoning_analysis.json"
            )
            if default_analysis.exists():
                args.analysis_json = str(default_analysis)
            else:
                logging.error("No --analysis_json provided and default not found.")
                return

        logging.info(f"Loading analysis from {args.analysis_json}")
        with open(args.analysis_json) as f:
            analysis = json.load(f)

        if target_correct_ids is None:
            target_correct_ids = [
                s["id"] for s in analysis.get("selected_correct_samples", [])
            ]
        if target_incorrect_ids is None:
            target_incorrect_ids = [
                s["id"] for s in analysis.get("selected_incorrect_samples", [])
            ]

    logging.info(f"Correct sample IDs:   {target_correct_ids}")
    logging.info(f"Incorrect sample IDs: {target_incorrect_ids}")

    model, processor = load_model(args.model_path)
    num_layers = get_num_layers(model)

    n_sample_layers = min(args.num_layers, num_layers)
    layers_to_capture = sorted(set(
        [int(i * (num_layers - 1) / max(n_sample_layers - 1, 1))
         for i in range(n_sample_layers)]
    ))
    logging.info(f"Model has {num_layers} layers, capturing: {layers_to_capture}")

    correct_vis_data = []
    incorrect_vis_data = []

    all_target_ids = (
        [(sid, True) for sid in target_correct_ids]
        + [(sid, False) for sid in target_incorrect_ids]
    )

    for sid, is_correct in all_target_ids:
        item = id_to_item.get(sid)
        if item is None:
            logging.warning(f"Sample ID {sid} not found in CVBench.json, skipping.")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Processing sample #{sid} ({'correct' if is_correct else 'incorrect'})")
        logging.info(f"  Task: {item.get('task_type', '?')}")
        
        # 显示完整的 Query，不再限制到 80 个字符
        logging.info(f"  Q: {item['question']}")

        video_frames = {}
        for vkey in ["video_1", "video_2", "video_3", "video_4"]:
            vrel = item.get(vkey)
            if vrel is None:
                continue
            vpath = str(data_dir / vrel)
            if not os.path.exists(vpath):
                logging.warning(f"  Video not found: {vpath}")
                continue
            frames = extract_frames(vpath, args.num_frames)
            if frames:
                video_frames[vrel] = frames
                logging.info(f"  {vkey}: {len(frames)} frames")

        if not video_frames:
            logging.warning("  No frames, skipping.")
            continue

        messages, all_images, video_labels = build_prompt(item, video_frames)

        try:
            gen_text, attn_data = inference_with_attention(
                model, processor, messages, all_images,
                max_new_tokens=args.max_new_tokens,
                layers_to_capture=layers_to_capture,
            )
        except Exception as e:
            logging.error(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        if "prediction" not in item:
            item["prediction"] = gen_text.strip().split()[0] if gen_text.strip() else "?"
        if "ground_truth" not in item:
            item["ground_truth"] = item.get("answer", "?")

        vis_path = str(output_dir / f"sample_{sid:04d}_{'correct' if is_correct else 'wrong'}.png")
        try:
            visualize_single_sample(
                item, video_frames, video_labels, attn_data, processor,
                vis_path, is_correct,
            )
        except Exception as e:
            logging.error(f"  Visualization error: {e}")
            import traceback
            traceback.print_exc()
            continue

        last_layer = max(attn_data["per_layer_attn"].keys())
        last_attn = attn_data["per_layer_attn"][last_layer]
        video_fracs, _ = compute_per_video_attention(
            last_attn, attn_data["image_token_mask"],
            video_labels, attn_data["input_len"],
        )
        layer_evenness = compute_cross_video_ratio_by_layer(attn_data, video_labels)

        summary = {
            "id": sid,
            "task_type": item.get("task_type", "?"),
            "pred": item.get("prediction", "?"),
            "gt": item.get("ground_truth", item.get("answer", "?")),
            "video_fracs": video_fracs,
            "layer_evenness": layer_evenness,
        }

        if is_correct:
            correct_vis_data.append(summary)
        else:
            incorrect_vis_data.append(summary)

        del attn_data
        torch.cuda.empty_cache()

    if correct_vis_data and incorrect_vis_data:
        comp_path = str(output_dir / "comparison_correct_vs_incorrect.png")
        visualize_comparison(correct_vis_data, incorrect_vis_data, comp_path)

    def make_serializable(d):
        if isinstance(d, dict):
            return {str(k): make_serializable(v) for k, v in d.items()}
        if isinstance(d, list):
            return [make_serializable(v) for v in d]
        if isinstance(d, (np.floating, float)):
            return round(float(d), 6)
        if isinstance(d, (np.integer, int)):
            return int(d)
        return d

    summary_data = {
        "correct_samples": make_serializable(correct_vis_data),
        "incorrect_samples": make_serializable(incorrect_vis_data),
    }
    summary_path = output_dir / "attention_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    logging.info(f"\nAttention summary saved to {summary_path}")

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Per-sample figures : {output_dir}/sample_*.png")
    print(f"Comparison figure  : {output_dir}/comparison_correct_vs_incorrect.png")
    print(f"Numerical summary  : {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()