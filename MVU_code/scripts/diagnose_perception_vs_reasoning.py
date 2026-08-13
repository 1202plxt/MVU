#!/usr/bin/env python
"""
CVBench Two-Stage Diagnostic: Perception vs Reasoning Error Analysis

The hypothesis: the model misperceives video content at the observation stage,
then constructs plausible-sounding but wrong reasoning on top of incorrect
observations. This script separates the two stages to find where errors occur.

== STAGE 1: Perception Probe ==
For each video in a sample, ask the model independently:
  "Describe what you see in this video in detail."
Then ask targeted perception questions relevant to the CVBench task
(e.g., "What objects appear?", "What action is being performed?").

== STAGE 2: Text-Only Reasoning ==
Feed the model's OWN descriptions from Stage 1 as pure text (no images),
along with the original question. If the model answers correctly from its
own descriptions, the error was in perception. If it still fails, the
error is in reasoning.

== STAGE 3: Gold Description Reasoning ==
Feed human-written (or GPT-generated) accurate descriptions + the question.
This is the ceiling: if the model still fails, the task itself is beyond
its reasoning capacity regardless of perception quality.

Outputs a diagnostic JSON classifying each error as:
  - "perception_error":  wrong description → wrong answer, but correct
                          answer when given accurate text descriptions
  - "reasoning_error":   correct description → still wrong answer
  - "both":              wrong description AND wrong reasoning from text
  - "perception_noise":  slightly off description but still got right answer

Usage:
    python scripts/diagnose_perception_vs_reasoning.py \
        --model_path /fs0/AI/sx2624011/LKOPD/MVU/shared_models/Qwen3-VL-8B-Instruct \
        --data_dir /fs0/AI/sx2624011/LKOPD/MVU/MVU_data/datasets/CVBench \
        --analysis_json /fs0/AI/sx2624011/LKOPD/MVU/MVU_data/outputs/cvbench_reasoning/reasoning_analysis.json \
        --output_dir /fs0/AI/sx2624011/LKOPD/MVU/MVU_data/outputs/cvbench_diagnosis \
        --num_frames 4

    # Or specify sample IDs directly:
    python scripts/diagnose_perception_vs_reasoning.py \
        --sample_ids 3 12 27 45 78 91 \
        --num_frames 4
"""

import os
import json
import re
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
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        model_cls = Qwen2_5_VLForConditionalGeneration

    logging.info(f"Loading model from {model_path}")
    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Resolution is configured separately via load_model_with_resolution()
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    return model, processor


# Qwen-VL resolution presets:
# Each frame is resized to fit within [min_pixels, max_pixels], then split
# into 28x28 patches. Each patch = 1 visual token.
#
# Preset         min_pixels   max_pixels   ~tokens/frame   notes
# ─────────────────────────────────────────────────────────────────
# low            64*28*28     128*28*28     64~128         最省显存
# medium        128*28*28     256*28*28    128~256         默认(原设置)
# high          256*28*28     512*28*28    256~512         清晰但费显存
# max           512*28*28    1280*28*28    512~1280        Qwen默认,单视频用
#
# 显存估算 (bf16, 8B模型本身~16GB):
#   total_visual_tokens = num_videos × num_frames × tokens_per_frame
#   例: 2视频 × 8帧 × 256token = 4096 visual tokens → ~20GB额外显存
#       2视频 × 16帧 × 128token = 4096 visual tokens → 类似
#       2视频 × 4帧 × 128token = 1024 visual tokens → ~5GB额外显存

RESOLUTION_PRESETS = {
    "low":    (64  * 28 * 28,  128 * 28 * 28),   # 省显存,适合多帧
    "medium": (128 * 28 * 28,  256 * 28 * 28),   # 平衡
    "high":   (256 * 28 * 28,  512 * 28 * 28),   # 高清,少帧
    "max":    (512 * 28 * 28, 1280 * 28 * 28),   # Qwen默认,单视频
}


def configure_processor(processor, resolution: str = "medium"):
    """Apply resolution preset to the processor."""
    if resolution in RESOLUTION_PRESETS:
        min_px, max_px = RESOLUTION_PRESETS[resolution]
    else:
        # Try parsing as "min-max", e.g. "100000-200000"
        try:
            parts = resolution.split("-")
            min_px, max_px = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            logging.warning(f"Unknown resolution '{resolution}', using 'medium'")
            min_px, max_px = RESOLUTION_PRESETS["medium"]

    processor.image_processor.min_pixels = min_px
    processor.image_processor.max_pixels = max_px

    tokens_per_frame_min = min_px // (28 * 28)
    tokens_per_frame_max = max_px // (28 * 28)
    logging.info(
        f"Resolution: min_pixels={min_px} max_pixels={max_px} "
        f"(~{tokens_per_frame_min}-{tokens_per_frame_max} tokens/frame)"
    )
    return processor


# ═══════════════════════════════════════════════════════════════════════════
# Generic inference helper
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(
    model, processor, messages, images=None,
    max_new_tokens=1024, enable_thinking=False,
) -> str:
    """
    Run inference with configurable token limit and optional thinking mode.
    Returns the clean decoded text (skip_special_tokens=True).
    """
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **({"enable_thinking": True} if enable_thinking else {}),
        )
    except TypeError:
        # Older transformers without enable_thinking support
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated[0, input_len:]

    # Return both full (with special tokens) and clean text
    full_text = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
    clean_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    # Log if output was likely truncated (hit max_new_tokens)
    if output_ids.shape[0] >= max_new_tokens:
        logging.warning(
            f"    Output hit max_new_tokens={max_new_tokens} "
            f"({output_ids.shape[0]} tokens) — may be truncated!"
        )

    return clean_text


# ═══════════════════════════════════════════════════════════════════════════
# Task-specific perception probes
# ═══════════════════════════════════════════════════════════════════════════

# CVBench task types and what perception questions matter for each
TASK_PERCEPTION_QUESTIONS = {
    "object_association": [
        "List all distinct objects you can see in this video.",
        "Describe the main subject and background of this video.",
        "Are there any text, logos, or identifiable brands visible?",
    ],
    "event_association": [
        "Describe the sequence of events or actions in this video, in order.",
        "What is the main activity happening? When does it start and end?",
        "Are there any notable changes in scene or setting during the video?",
    ],
    "complex_reasoning": [
        "Describe the setting, environment, and context of this video.",
        "What people, objects, or activities are visible?",
        "Are there any clues about time of day, season, location, or event type?",
    ],
}

# Fallback for unknown task types
DEFAULT_PERCEPTION_QUESTIONS = [
    "Describe everything you see in this video in detail: objects, actions, setting, and any text or notable features.",
]


def get_perception_questions(task_type: str) -> List[str]:
    """Return perception probe questions appropriate for the task type."""
    # Normalize task_type
    normalized = task_type.lower().replace(" ", "_").replace("-", "_")
    for key, questions in TASK_PERCEPTION_QUESTIONS.items():
        if key in normalized:
            return questions
    return DEFAULT_PERCEPTION_QUESTIONS


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: Per-video perception probe
# ═══════════════════════════════════════════════════════════════════════════
def stage1_perception_probe(
    model, processor, item: dict, video_frames: Dict[str, List[Image.Image]],
    num_frames: int, data_dir: Path,
    max_new_tokens: int = 1024, enable_thinking: bool = False,
) -> Dict[str, dict]:
    """
    For each video in the sample, independently ask the model to describe
    what it sees. Also ask task-specific perception questions.

    Returns dict[video_key] -> {
        "description": str,
        "perception_qa": [ {"question": str, "answer": str}, ... ],
    }
    """
    task_type = item.get("task_type", "unknown")
    probe_questions = get_perception_questions(task_type)

    # Also provide question context so the model focuses on relevant details
    question_context = item["question"]

    results = {}
    for vkey in ["video_1", "video_2", "video_3", "video_4"]:
        vrel = item.get(vkey)
        if vrel is None or vrel not in video_frames:
            continue

        frames = video_frames[vrel]
        if not frames:
            continue

        logging.info(f"    Probing {vkey} ({len(frames)} frames)...")

        # ── General description (question-guided) ──
        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({
            "type": "text",
            "text": (
                f"These {len(frames)} frames are uniformly sampled from a video.\n\n"
                f"Context: You will later need to answer this question about "
                f"multiple videos: \"{question_context}\"\n\n"
                f"For now, focus ONLY on THIS video. Describe in detail:\n"
                f"1. What objects, people, or animals are visible?\n"
                f"2. What actions or events are happening, in what order?\n"
                f"3. What is the setting/environment?\n"
                f"4. Any text, numbers, colors, spatial relationships, or other "
                f"details that might be relevant to the question above.\n\n"
                f"Be specific, factual, and thorough. Describe ONLY what you "
                f"can actually see in the frames."
            ),
        })
        messages = [{"role": "user", "content": content}]
        description = run_inference(
            model, processor, messages, frames,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )

        # ── Task-specific perception questions ──
        # Ask as a single multi-part question to save inference calls
        combined_pq = "\n".join(f"- {pq}" for pq in probe_questions)
        content_q = []
        for frame in frames:
            content_q.append({"type": "image", "image": frame})
        content_q.append({
            "type": "text",
            "text": (
                f"These frames are from a video. Please answer each question "
                f"below based on what you see:\n\n{combined_pq}\n\n"
                f"Answer each question separately and thoroughly."
            ),
        })
        messages_q = [{"role": "user", "content": content_q}]
        combined_ans = run_inference(
            model, processor, messages_q, frames,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        pqa_results = [{"question": combined_pq, "answer": combined_ans}]

        results[vkey] = {
            "description": description,
            "perception_qa": pqa_results,
        }
        logging.info(f"    {vkey} desc ({len(description)} chars): {description[:100]}...")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: Text-only reasoning (using model's own descriptions)
# ═══════════════════════════════════════════════════════════════════════════
def stage2_text_only_reasoning(
    model, processor, item: dict,
    perception_results: Dict[str, dict],
    max_new_tokens: int = 1024, enable_thinking: bool = False,
) -> Tuple[str, str]:
    """
    Feed the model's own video descriptions as text + the original question.
    No images. Tests whether the reasoning is correct given what the model
    "thinks" it saw.

    Returns (predicted_answer, full_response).
    """
    # Build text prompt from descriptions
    desc_parts = []
    for vkey in ["video_1", "video_2", "video_3", "video_4"]:
        if vkey not in perception_results:
            continue
        desc = perception_results[vkey]["description"]
        # Also include perception QA answers for richer context
        pqa_text = ""
        for pqa in perception_results[vkey].get("perception_qa", []):
            pqa_text += f"\n  Additional observations: {pqa['answer']}"
        desc_parts.append(f"[{vkey} description]:\n{desc}{pqa_text}")

    descriptions_text = "\n\n".join(desc_parts)

    question = item["question"]
    options = item.get("options", [])
    opts_text = "\n".join(options) if options else ""

    if opts_text:
        prompt = (
            f"You are given detailed descriptions of multiple videos. Based on "
            f"these descriptions, answer the question.\n\n"
            f"{descriptions_text}\n\n"
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n\n"
            f"Think step by step:\n"
            f"1. What relevant information does each video description contain?\n"
            f"2. How do the videos compare or relate to each other?\n"
            f"3. Which option best answers the question?\n\n"
            f"State your answer as 'Final Answer: X' where X is the option letter."
        )
    else:
        prompt = (
            f"You are given detailed descriptions of multiple videos. Based on "
            f"these descriptions, answer the question.\n\n"
            f"{descriptions_text}\n\n"
            f"Question: {question}\n\n"
            f"Think step by step, then state your answer clearly as "
            f"'Final Answer: ...'."
        )

    messages = [{"role": "user", "content": prompt}]
    response = run_inference(
        model, processor, messages, images=None,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    pred = extract_answer(response, options)
    return pred, response


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: Direct multi-video answer (baseline, same as original eval)
# ═══════════════════════════════════════════════════════════════════════════
def stage3_direct_multivideo(
    model, processor, item: dict, video_frames: Dict[str, List[Image.Image]],
    max_new_tokens: int = 1024, enable_thinking: bool = False,
) -> Tuple[str, str]:
    """Standard multi-video inference as baseline."""
    content = []
    all_images = []
    for vkey in ["video_1", "video_2", "video_3", "video_4"]:
        vrel = item.get(vkey)
        if vrel is None or vrel not in video_frames:
            continue
        frames = video_frames[vrel]
        if not frames:
            continue
        content.append({"type": "text", "text": f"\n[{vkey}]:"})
        for frame in frames:
            content.append({"type": "image", "image": frame})
            all_images.append(frame)

    question = item["question"]
    options = item.get("options", [])
    opts_text = "\n".join(options) if options else ""
    if opts_text:
        q = (
            f"\n\nQuestion: {question}\nOptions:\n{opts_text}\n\n"
            "Think step by step:\n"
            "1. Describe what you observe in each video.\n"
            "2. Compare across videos.\n"
            "3. State 'Final Answer: X' where X is the option letter."
        )
    else:
        q = (
            f"\n\nQuestion: {question}\n\n"
            "Think step by step, then state 'Final Answer: ...'."
        )
    content.append({"type": "text", "text": q})

    messages = [{"role": "user", "content": content}]
    response = run_inference(
        model, processor, messages, all_images,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    pred = extract_answer(response, options)
    return pred, response


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: Single-video QA (isolation test)
# ═══════════════════════════════════════════════════════════════════════════
def stage4_single_video_qa(
    model, processor, item: dict, video_frames: Dict[str, List[Image.Image]],
    max_new_tokens: int = 1024, enable_thinking: bool = False,
) -> Dict[str, dict]:
    """
    For questions that reference specific videos (e.g., "which video shows X"),
    test whether the model can answer a simplified per-video question.

    This isolates whether confusion comes from having multiple videos in
    context simultaneously (interference) vs inability to understand each
    video alone.
    """
    question = item["question"]
    options = item.get("options", [])

    results = {}
    for vkey in ["video_1", "video_2", "video_3", "video_4"]:
        vrel = item.get(vkey)
        if vrel is None or vrel not in video_frames:
            continue
        frames = video_frames[vrel]
        if not frames:
            continue

        content = []
        for frame in frames:
            content.append({"type": "image", "image": frame})

        simplified_q = (
            f"This is {vkey}. You will later need to answer this question "
            f"about multiple videos:\n\"{question}\"\n\n"
            f"For now, focus ONLY on THIS video ({vkey}). Describe in detail:\n"
            f"- What objects, people, or animals appear?\n"
            f"- What actions or events happen?\n"
            f"- What details are relevant to the question above?\n\n"
            f"Be specific and thorough."
        )
        content.append({"type": "text", "text": simplified_q})

        messages = [{"role": "user", "content": content}]
        response = run_inference(
            model, processor, messages, frames,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        results[vkey] = {"response": response}
        logging.info(f"    {vkey} single-video ({len(response)} chars): {response[:100]}...")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Answer extraction (same as reasoning script)
# ═══════════════════════════════════════════════════════════════════════════
def extract_answer(text: str, options: list) -> str:
    text_clean = text.strip()
    if options and len(options) <= 2:
        upper = text_clean.upper()
        if "YES" in upper.split()[:3]:
            return "Yes"
        if "NO" in upper.split()[:3]:
            return "No"
        m = re.search(r"Final\s*Answer\s*[:：]\s*(Yes|No)", text_clean, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return text_clean.split()[0] if text_clean.split() else ""

    m = re.search(
        r"(?:Final\s*Answer|The\s*answer\s*is|Answer)\s*[:：]\s*([A-D])",
        text_clean, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    m2 = re.match(r"^([A-D])[\.\s,:\)]", text_clean, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    m3 = re.search(r"\b([A-D])\b", text_clean)
    if m3:
        return m3.group(1).upper()
    return text_clean.split()[0] if text_clean.split() else ""


def check_correct(pred: str, gt: str) -> bool:
    return pred.strip().upper().rstrip(".") == gt.strip().upper().rstrip(".")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnosis logic
# ═══════════════════════════════════════════════════════════════════════════
def diagnose_error(
    direct_correct: bool,
    text_only_correct: bool,
    perception_results: Dict[str, dict],
    single_video_results: Dict[str, dict],
) -> dict:
    """
    Classify the error source based on multi-stage results.

    Decision tree:
      direct_correct=True  → no error to diagnose
      direct_correct=False →
        text_only_correct=True  → "perception_error"
            (model can reason correctly from text, so the visual
             perception was the bottleneck)
        text_only_correct=False →
            Check if descriptions look reasonable...
            If descriptions are clearly wrong → "both"
            If descriptions seem ok → "reasoning_error"
    """
    if direct_correct:
        return {
            "diagnosis": "correct",
            "explanation": "Model answered correctly with direct multi-video input.",
        }

    if text_only_correct:
        return {
            "diagnosis": "perception_error",
            "explanation": (
                "Model answered WRONG with video frames but CORRECT when given "
                "its own text descriptions. This means the visual encoder or "
                "vision-language alignment is the bottleneck — the model can "
                "reason correctly but cannot perceive the video content accurately "
                "when processing raw frames in the multi-video context."
            ),
        }

    # Both wrong — need to figure out if descriptions were at least reasonable
    # We can't automatically judge description quality without ground truth,
    # so we flag for manual review and provide what we can
    return {
        "diagnosis": "reasoning_error_or_both",
        "explanation": (
            "Model answered WRONG both with video frames AND with its own text "
            "descriptions. Either: (1) the descriptions were wrong AND reasoning "
            "was wrong ('both'), or (2) the descriptions were roughly correct but "
            "the reasoning/comparison step failed ('reasoning_error'). "
            "Check the descriptions manually to distinguish."
        ),
        "needs_manual_review": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════
BG = "#1A1A2E"
CARD = "#16213E"
TXT = "#E0E0E0"
SUBTLE = "#8899AA"
GREEN = "#22C55E"
RED = "#EF4444"
YELLOW = "#FBBF24"
BLUE = "#3B82F6"

DIAG_COLORS = {
    "correct": GREEN,
    "perception_error": YELLOW,
    "reasoning_error_or_both": RED,
}


def visualize_diagnosis_summary(all_diagnoses: List[dict], save_path: str):
    """
    Summary figure:
      - Pie chart of error categories
      - Per-sample comparison table (direct vs text-only)
      - Flowchart showing the diagnostic logic
    """
    fig = plt.figure(figsize=(20, 12), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.94, top=0.90, bottom=0.06)

    fig.suptitle(
        "Perception vs Reasoning Error Diagnosis",
        fontsize=18, fontweight="bold", color=TXT, y=0.96,
    )

    # ── Count diagnoses ──
    counts = {}
    for d in all_diagnoses:
        diag = d["diagnosis"]["diagnosis"]
        counts[diag] = counts.get(diag, 0) + 1

    # ── Pie chart ──
    ax_pie = fig.add_subplot(gs[0, 0], facecolor=CARD)
    if counts:
        labels = list(counts.keys())
        sizes = [counts[l] for l in labels]
        colors = [DIAG_COLORS.get(l, SUBTLE) for l in labels]
        display_labels = [
            l.replace("_", " ").title() for l in labels
        ]
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=display_labels, colors=colors,
            autopct="%1.0f%%", textprops={"fontsize": 10, "color": TXT},
            wedgeprops={"edgecolor": BG, "linewidth": 2},
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
    ax_pie.set_title("Error Category Distribution", fontsize=13,
                     fontweight="bold", color=TXT, pad=15)

    # ── Bar chart: direct vs text-only accuracy ──
    ax_bar = fig.add_subplot(gs[0, 1], facecolor=CARD)
    n_total = len(all_diagnoses)
    direct_correct = sum(1 for d in all_diagnoses if d["direct_correct"])
    text_correct = sum(1 for d in all_diagnoses if d["text_only_correct"])
    single_vid_helpful = sum(
        1 for d in all_diagnoses
        if d.get("single_video_adds_info", False)
    )

    bar_labels = [
        "Direct\nMulti-Video",
        "Text-Only\n(Own Descriptions)",
    ]
    bar_vals = [
        direct_correct / n_total * 100 if n_total else 0,
        text_correct / n_total * 100 if n_total else 0,
    ]
    bar_colors = [BLUE, GREEN]

    bars = ax_bar.bar(bar_labels, bar_vals, color=bar_colors, width=0.5,
                      edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, bar_vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}%", ha="center", fontsize=12, fontweight="bold",
                    color=TXT)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Accuracy (%)", fontsize=11, color=TXT)
    ax_bar.set_title("Accuracy: Direct vs Text-Only", fontsize=13,
                     fontweight="bold", color=TXT, pad=15)
    ax_bar.tick_params(colors=TXT)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for s in ["bottom", "left"]:
        ax_bar.spines[s].set_color(SUBTLE)

    # ── Per-sample results table ──
    ax_table = fig.add_subplot(gs[1, :], facecolor=CARD)
    ax_table.axis("off")

    col_labels = [
        "Sample ID", "Task Type", "GT",
        "Direct\nPred", "Direct\nResult",
        "Text-Only\nPred", "Text-Only\nResult",
        "Diagnosis",
    ]
    table_data = []
    cell_colors = []

    for d in all_diagnoses:
        row = [
            str(d["id"]),
            d["task_type"][:20],
            d["gt"],
            d["direct_pred"],
            "✓" if d["direct_correct"] else "✗",
            d["text_only_pred"],
            "✓" if d["text_only_correct"] else "✗",
            d["diagnosis"]["diagnosis"].replace("_", "\n"),
        ]
        table_data.append(row)

        # Row colors
        row_colors = [CARD] * len(col_labels)
        diag = d["diagnosis"]["diagnosis"]
        dc = DIAG_COLORS.get(diag, SUBTLE)
        # Highlight result columns
        row_colors[4] = GREEN + "40" if d["direct_correct"] else RED + "40"
        row_colors[6] = GREEN + "40" if d["text_only_correct"] else RED + "40"
        row_colors[7] = dc + "60"
        cell_colors.append(row_colors)

    if table_data:
        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)

        # Style
        for key, cell in table.get_celld().items():
            cell.set_edgecolor(SUBTLE)
            cell.set_linewidth(0.5)
            if key[0] == 0:
                # Header
                cell.set_facecolor("#0F3460")
                cell.set_text_props(color=TXT, fontweight="bold")
            else:
                cell.set_facecolor(CARD)
                cell.set_text_props(color=TXT)

    ax_table.set_title(
        "Per-Sample Diagnostic Results",
        fontsize=13, fontweight="bold", color=TXT, pad=20,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    logging.info(f"Diagnosis summary saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="CVBench Perception vs Reasoning Diagnosis"
    )
    parser.add_argument("--model_path", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/shared_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/datasets/CVBench")
    parser.add_argument("--analysis_json", type=str, default="",
                        help="Path to reasoning_analysis.json (to pick samples)")
    parser.add_argument("--output_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/outputs/cvbench_diagnosis")
    parser.add_argument("--sample_ids", type=int, nargs="*", default=None,
                        help="Manually specify sample IDs to diagnose")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="If no sample_ids, evaluate first N and diagnose errors")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames per video (default 8, use 4 if OOM)")
    parser.add_argument("--resolution", type=str, default="medium",
                        help="Resolution preset: low/medium/high/max, or 'min-max' pixels (e.g. '100000-200000')")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate per call (default 1024, use 2048 for thinking mode)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3-VL native <think> mode for richer output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    json_path = data_dir / "CVBench.json"
    with open(json_path) as f:
        all_data = json.load(f)
    id_to_item = {item.get("id", i): item for i, item in enumerate(all_data)}

    # ── Determine sample IDs ──
    if args.sample_ids:
        target_ids = args.sample_ids
    elif args.analysis_json:
        with open(args.analysis_json) as f:
            analysis = json.load(f)
        # Take all selected samples (both correct and incorrect)
        target_ids = (
            [s["id"] for s in analysis.get("selected_correct_samples", [])]
            + [s["id"] for s in analysis.get("selected_incorrect_samples", [])]
        )
    else:
        # Run on first N samples
        target_ids = [item.get("id", i) for i, item in enumerate(all_data[:args.num_samples])]

    logging.info(f"Diagnosing {len(target_ids)} samples: {target_ids}")
    logging.info(f"Config: max_new_tokens={args.max_new_tokens}, "
                 f"enable_thinking={args.enable_thinking}, "
                 f"num_frames={args.num_frames}")

    # ── Load model ──
    model, processor = load_model(args.model_path)
    processor = configure_processor(processor, args.resolution)

    # ── Token budget estimate ──
    # Count how many videos per sample (rough: check first target)
    sample_item = id_to_item.get(target_ids[0]) if target_ids else None
    if sample_item:
        n_vids = sum(1 for vk in ["video_1","video_2","video_3","video_4"]
                     if sample_item.get(vk) is not None)
        preset = RESOLUTION_PRESETS.get(args.resolution)
        if preset:
            max_tok_per_frame = preset[1] // (28 * 28)
        else:
            max_tok_per_frame = 256
        total_visual = n_vids * args.num_frames * max_tok_per_frame
        logging.info(
            f"Token budget estimate: {n_vids} videos × {args.num_frames} frames "
            f"× ~{max_tok_per_frame} tokens/frame = ~{total_visual} visual tokens"
        )
        if total_visual > 8000:
            logging.warning(
                f"  ⚠ {total_visual} visual tokens is HIGH — may OOM on 40GB GPU. "
                f"Consider: --num_frames {max(2, args.num_frames//2)} or --resolution low"
            )

    # ── Run diagnosis ──
    all_diagnoses = []

    for sid in target_ids:
        item = id_to_item.get(sid)
        if item is None:
            logging.warning(f"Sample {sid} not found, skipping.")
            continue

        gt = item.get("answer", "")
        task_type = item.get("task_type", "unknown")

        logging.info(f"\n{'═'*60}")
        logging.info(f"SAMPLE #{sid}  task={task_type}")
        logging.info(f"  Q: {item['question']}")
        logging.info(f"  GT: {gt}")

        # Load video frames
        video_frames = {}
        for vkey in ["video_1", "video_2", "video_3", "video_4"]:
            vrel = item.get(vkey)
            if vrel is None:
                continue
            vpath = str(data_dir / vrel)
            if not os.path.exists(vpath):
                continue
            frames = extract_frames(vpath, args.num_frames)
            if frames:
                video_frames[vrel] = frames

        if not video_frames:
            logging.warning("  No frames, skipping.")
            continue

        mnt = args.max_new_tokens
        etk = args.enable_thinking

        # ── STAGE 3 (baseline): Direct multi-video ──
        logging.info("  [Stage: Direct Multi-Video]")
        try:
            direct_pred, direct_response = stage3_direct_multivideo(
                model, processor, item, video_frames,
                max_new_tokens=mnt, enable_thinking=etk,
            )
        except Exception as e:
            logging.error(f"  Direct inference error: {e}")
            direct_pred, direct_response = "", f"ERROR: {e}"
        direct_correct = check_correct(direct_pred, gt)
        logging.info(f"  Direct: pred={direct_pred} correct={direct_correct}")
        logging.info(f"  Direct response ({len(direct_response)} chars): {direct_response[:150]}")

        # ── STAGE 1: Perception probe ──
        logging.info("  [Stage 1: Perception Probe]")
        try:
            perception_results = stage1_perception_probe(
                model, processor, item, video_frames, args.num_frames, data_dir,
                max_new_tokens=mnt, enable_thinking=etk,
            )
        except Exception as e:
            logging.error(f"  Perception probe error: {e}")
            perception_results = {}

        # ── STAGE 2: Text-only reasoning ──
        logging.info("  [Stage 2: Text-Only Reasoning]")
        try:
            text_pred, text_response = stage2_text_only_reasoning(
                model, processor, item, perception_results,
                max_new_tokens=mnt, enable_thinking=etk,
            )
        except Exception as e:
            logging.error(f"  Text-only error: {e}")
            text_pred, text_response = "", f"ERROR: {e}"
        text_correct = check_correct(text_pred, gt)
        logging.info(f"  Text-only: pred={text_pred} correct={text_correct}")
        logging.info(f"  Text-only response ({len(text_response)} chars): {text_response[:150]}")

        # ── STAGE 4: Single-video isolation ──
        logging.info("  [Stage 4: Single-Video QA]")
        try:
            single_video_results = stage4_single_video_qa(
                model, processor, item, video_frames,
                max_new_tokens=mnt, enable_thinking=etk,
            )
        except Exception as e:
            logging.error(f"  Single-video error: {e}")
            single_video_results = {}

        # ── Diagnose ──
        diagnosis = diagnose_error(
            direct_correct, text_correct,
            perception_results, single_video_results,
        )
        logging.info(f"  DIAGNOSIS: {diagnosis['diagnosis']}")

        entry = {
            "id": sid,
            "task_type": task_type,
            "question": item["question"],
            "options": item.get("options", []),
            "gt": gt,
            # Stage results
            "direct_pred": direct_pred,
            "direct_correct": direct_correct,
            "direct_response": direct_response,
            "text_only_pred": text_pred,
            "text_only_correct": text_correct,
            "text_only_response": text_response,
            # Perception details
            "perception_descriptions": {
                vkey: pr["description"]
                for vkey, pr in perception_results.items()
            },
            "perception_qa": {
                vkey: pr["perception_qa"]
                for vkey, pr in perception_results.items()
            },
            # Single-video isolation
            "single_video_responses": {
                vkey: sv["response"]
                for vkey, sv in single_video_results.items()
            },
            # Diagnosis
            "diagnosis": diagnosis,
        }
        all_diagnoses.append(entry)

    # ── Save full results ──
    results_path = output_dir / "diagnosis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_diagnoses, f, indent=2, ensure_ascii=False)
    logging.info(f"\nFull results saved to {results_path}")

    # ── Visualization ──
    vis_path = str(output_dir / "diagnosis_summary.png")
    visualize_diagnosis_summary(all_diagnoses, vis_path)

    # ── Console summary ──
    print("\n" + "=" * 60)
    print("PERCEPTION vs REASONING DIAGNOSIS SUMMARY")
    print("=" * 60)

    n_total = len(all_diagnoses)
    n_direct_correct = sum(1 for d in all_diagnoses if d["direct_correct"])
    n_text_correct = sum(1 for d in all_diagnoses if d["text_only_correct"])

    counts = {}
    for d in all_diagnoses:
        diag = d["diagnosis"]["diagnosis"]
        counts[diag] = counts.get(diag, 0) + 1

    print(f"Samples diagnosed  : {n_total}")
    print(f"Direct accuracy    : {n_direct_correct}/{n_total} = {n_direct_correct/n_total*100:.1f}%")
    print(f"Text-only accuracy : {n_text_correct}/{n_total} = {n_text_correct/n_total*100:.1f}%")
    print()
    print("Error breakdown:")
    for diag, count in sorted(counts.items()):
        pct = count / n_total * 100
        print(f"  {diag:30s}: {count:3d} ({pct:.1f}%)")

    print()
    if n_text_correct > n_direct_correct:
        gap = n_text_correct - n_direct_correct
        print(f"KEY FINDING: {gap} samples answered correctly from text descriptions")
        print(f"  but WRONG from video frames → these are PERCEPTION ERRORS.")
        print(f"  The visual encoder / vision-language alignment is the bottleneck.")
    elif n_text_correct == n_direct_correct:
        print("TEXT-ONLY accuracy matches DIRECT accuracy → errors are likely")
        print("  in REASONING, not perception (or descriptions are also wrong).")
    else:
        print("Interesting: DIRECT accuracy > TEXT-ONLY accuracy.")
        print("  Some visual information is lost when converted to text descriptions.")

    print(f"\nDetailed results: {results_path}")
    print(f"Summary figure:   {vis_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()