#!/usr/bin/env python
"""
CVBench Reasoning Analysis for Qwen3-VL-8B-Instruct

Evaluates Qwen3-VL on CVBench, captures the model's full reasoning process,
selects 3 correct + 3 incorrect samples, and analyzes whether the model
reasons before answering or answers before explaining.

Usage:
    python MVU/MVU_code/scripts/eval_cvbench_reasoning.py \
        --model_path MVU/shared_models/Qwen3-VL-8B-Instruct \
        --data_dir   MVU/MVU_data/datasets/CVBench \
        --output_dir MVU/MVU_data/outputs/cvbench_reasoning \
        --num_samples 50 \
        --num_frames 4
"""

import os
import json
import re
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from PIL import Image


# ── Video frame extraction ──────────────────────────────────────────────────
def extract_frames_from_video(video_path: str, num_frames: int = 8) -> List[Image.Image]:
    """Uniformly sample *num_frames* from a video file and return as PIL Images."""
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


# ── Model loading ───────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load Qwen3-VL model and processor with eager attention."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
        model_cls = Qwen3VLForConditionalGeneration
    except ImportError:
        model_cls = Qwen2_5_VLForConditionalGeneration

    logging.info(f"Loading model from {model_path} ...")
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
    logging.info("Model loaded.")
    return model, processor


# ── Build prompt (with CoT instruction) ─────────────────────────────────────
def build_prompt(
    item: dict,
    video_frames: Dict[str, List[Image.Image]],
    enable_cot: bool = True,
) -> Tuple[list, list, list]:
    """
    Build a chat-style prompt for Qwen3-VL.

    When enable_cot=True, the prompt explicitly asks the model to show its
    reasoning process step by step BEFORE giving the final answer.  This lets
    us capture and later analyse the chain-of-thought.

    Returns (messages, all_images_flat, video_labels).
    """
    content_parts = []
    all_images: List[Image.Image] = []
    video_labels: List[str] = []

    # Add each video's frames
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

    # Build question text
    question = item["question"]
    options = item.get("options", [])
    if options:
        opts_text = "\n".join(options)
    else:
        opts_text = ""

    if enable_cot:
        # Prompt designed to elicit structured reasoning
        if opts_text:
            question_text = (
                f"\n\nQuestion: {question}\n"
                f"Options:\n{opts_text}\n\n"
                "Please think step by step:\n"
                "1. First, describe what you observe in each video.\n"
                "2. Then, compare or relate the information across videos.\n"
                "3. Finally, state your answer as 'Final Answer: X' where X is "
                "the option letter (A/B/C/D) or Yes/No.\n"
            )
        else:
            question_text = (
                f"\n\nQuestion: {question}\n\n"
                "Please think step by step:\n"
                "1. First, describe what you observe in each video.\n"
                "2. Then, reason about the question.\n"
                "3. Finally, state your answer clearly.\n"
            )
    else:
        # Direct answering without CoT
        if opts_text:
            question_text = (
                f"\n\nQuestion: {question}\n"
                f"Options:\n{opts_text}\n\n"
                "Answer with the correct option letter and explain briefly."
            )
        else:
            question_text = (
                f"\n\nQuestion: {question}\n"
                "Please answer briefly."
            )

    content_parts.append({"type": "text", "text": question_text})
    messages = [{"role": "user", "content": content_parts}]
    return messages, all_images, video_labels


# ── Inference ───────────────────────────────────────────────────────────────
@torch.no_grad()
def inference(
    model,
    processor,
    messages,
    all_images,
    max_new_tokens: int = 512,
    enable_thinking: bool = False,
):
    """
    Run generation.

    If enable_thinking=True and the model supports it (Qwen3-VL with
    thinking mode), we try to capture <think>...</think> blocks.
    Otherwise we rely on prompt-based CoT.

    Returns the full generated text (including any <think> block).
    """
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3-VL-specific: enable_thinking triggers <think> block
        **({"enable_thinking": True} if enable_thinking else {}),
    )

    inputs = processor(
        text=[text],
        images=all_images if all_images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]
    logging.info(f"  Input length: {input_len} tokens")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    output_ids = generated_ids[0, input_len:]
    # Decode WITHOUT skipping special tokens so we can see <think>...</think>
    full_text = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
    clean_text = processor.tokenizer.decode(output_ids, skip_special_tokens=True)

    logging.info(f"  Generated {output_ids.shape[0]} tokens")
    return full_text, clean_text


# ── Reasoning pattern analysis ──────────────────────────────────────────────

def parse_thinking_block(full_text: str) -> Tuple[str, str]:
    """
    Separate <think>...</think> block from the visible answer.
    Returns (thinking_text, answer_text).
    """
    think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        # Everything after </think> is the visible answer
        after = full_text[think_match.end():].strip()
        # Clean residual special tokens
        after = re.sub(r"<\|[^>]+\|>", "", after).strip()
        return thinking, after
    return "", full_text


def analyse_reasoning_pattern(full_text: str, clean_text: str) -> dict:
    """
    Determine whether the model:
      (A) Reasons first, then gives the answer  ("reason-first")
      (B) States the answer first, then explains ("answer-first")
      (C) Uses <think> block (native thinking mode) ("thinking-mode")
      (D) Gives answer only, no reasoning          ("answer-only")

    Returns a dict with:
      - pattern: one of the above labels
      - thinking_text: content of <think> block (if any)
      - answer_text: the visible answer portion
      - final_answer_location: character offset of the final answer in clean_text
      - reasoning_text: the reasoning portion (from CoT or thinking block)
    """
    result = {
        "pattern": "unknown",
        "thinking_text": "",
        "answer_text": "",
        "reasoning_text": "",
        "final_answer_location": -1,
        "answer_appears_before_reasoning": False,
    }

    # ── Check for native <think> block ──
    thinking, after_think = parse_thinking_block(full_text)
    if thinking:
        result["pattern"] = "thinking-mode"
        result["thinking_text"] = thinking
        result["answer_text"] = after_think
        result["reasoning_text"] = thinking
        return result

    # ── Analyse the clean text for CoT patterns ──
    text = clean_text.strip()
    result["answer_text"] = text

    # Look for explicit "Final Answer:" marker
    final_match = re.search(
        r"(?:Final\s*Answer|The\s*answer\s*is|Answer)\s*[:：]\s*([A-D]|Yes|No)",
        text,
        re.IGNORECASE,
    )

    # Look for an answer letter at the very start
    starts_with_answer = bool(re.match(r"^[A-D][\.\s,:\)]", text))

    # Look for reasoning indicators
    reasoning_indicators = [
        r"(?:let|let's)\s+(?:me\s+)?(?:think|analyze|look|examine|consider)",
        r"(?:first|step\s*1|1[\.\)])\s*[,:]?\s*(?:I|we|let|in|the|looking)",
        r"(?:in|from)\s+video[\s_]?\d",
        r"looking at",
        r"comparing",
        r"observ(?:e|ing)",
    ]
    has_reasoning = any(
        re.search(p, text, re.IGNORECASE) for p in reasoning_indicators
    )

    # Minimal length heuristic: very short = answer-only
    word_count = len(text.split())

    if word_count <= 5:
        result["pattern"] = "answer-only"
        result["reasoning_text"] = ""
    elif final_match:
        result["final_answer_location"] = final_match.start()
        reasoning_before = text[: final_match.start()].strip()
        if len(reasoning_before.split()) > 5:
            result["pattern"] = "reason-first"
            result["reasoning_text"] = reasoning_before
        else:
            result["pattern"] = "answer-first"
            result["reasoning_text"] = text[final_match.end():].strip()
    elif starts_with_answer:
        result["pattern"] = "answer-first"
        result["answer_appears_before_reasoning"] = True
        result["reasoning_text"] = text[2:].strip()  # after "A." etc.
    elif has_reasoning:
        result["pattern"] = "reason-first"
        result["reasoning_text"] = text
    else:
        # Default: treat short outputs as answer-only, longer as reason-first
        result["pattern"] = "reason-first" if word_count > 15 else "answer-only"
        result["reasoning_text"] = text if word_count > 15 else ""

    return result


# ── Answer extraction ───────────────────────────────────────────────────────
def extract_answer(text: str, options: list) -> str:
    """Extract the predicted answer letter from generated text."""
    text_clean = text.strip()

    # Yes/No questions
    if options and len(options) <= 2:
        upper = text_clean.upper()
        if "YES" in upper.split()[:3]:
            return "Yes"
        if "NO" in upper.split()[:3]:
            return "No"
        # Check for Final Answer
        m = re.search(r"Final\s*Answer\s*[:：]\s*(Yes|No)", text_clean, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return text_clean.split()[0] if text_clean.split() else ""

    # Multiple choice A/B/C/D
    # Priority 1: "Final Answer: X"
    m = re.search(r"(?:Final\s*Answer|The\s*answer\s*is|Answer)\s*[:：]\s*([A-D])", text_clean, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Priority 2: starts with answer letter
    m2 = re.match(r"^([A-D])[\.\s,:\)]", text_clean, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    # Priority 3: first standalone A/B/C/D
    m3 = re.search(r"\b([A-D])\b", text_clean)
    if m3:
        return m3.group(1).upper()

    return text_clean.split()[0] if text_clean.split() else ""


def check_correct(pred: str, gt: str) -> bool:
    pred_c = pred.strip().upper().rstrip(".")
    gt_c = gt.strip().upper().rstrip(".")
    return pred_c == gt_c


# ── Error categorisation ───────────────────────────────────────────────────
def categorise_error(reasoning_analysis: dict, item: dict) -> str:
    """
    Attempt a rough categorisation of why the model got it wrong based on
    the reasoning text.

    Categories:
      - "no_reasoning":       model gave almost no reasoning
      - "wrong_observation":  model described video content incorrectly
      - "correct_obs_wrong_reasoning": observations seem ok but conclusion wrong
      - "cross_video_confusion": model mixed up which video is which
      - "insufficient_comparison": model didn't compare across videos
      - "unknown":            can't determine
    """
    reasoning = reasoning_analysis.get("reasoning_text", "")
    if not reasoning or len(reasoning.split()) < 5:
        return "no_reasoning"

    lower = reasoning.lower()

    # Check for cross-video confusion signals
    # e.g., describing video_1 content when talking about video_2
    video_mentions = re.findall(r"video[\s_]?(\d)", lower)
    if len(set(video_mentions)) < 2 and item.get("video_2"):
        return "insufficient_comparison"

    # Check if model mentions comparing
    comparison_words = ["compar", "differ", "similar", "contrast", "both", "whereas", "while"]
    has_comparison = any(w in lower for w in comparison_words)
    if not has_comparison and item.get("video_2"):
        return "insufficient_comparison"

    # Check confusion patterns
    confusion_patterns = [
        r"video[\s_]?1.*(?:shows?|has|contains?).*video[\s_]?2",  # mixing descriptions
        r"(?:confus|mix|swap)",
    ]
    for p in confusion_patterns:
        if re.search(p, lower):
            return "cross_video_confusion"

    return "unknown"


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="CVBench Reasoning Analysis"
    )
    parser.add_argument("--model_path", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/shared_models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/datasets/CVBench")
    parser.add_argument("--output_dir", type=str,
                        default="/fs0/AI/sx2624011/LKOPD/MVU/MVU_data/outputs/cvbench_reasoning")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate (-1 = all)")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Frames per video (4 for 40GB GPU, 8 for 80GB)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate (longer to capture full reasoning)")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Try to use Qwen3-VL native <think> mode")
    parser.add_argument("--num_select", type=int, default=10,
                        help="Number of correct/incorrect samples to select")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    json_path = data_dir / "CVBench.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    logging.info(f"Loading CVBench from {json_path}")
    with open(json_path) as f:
        dataset = json.load(f)
    logging.info(f"Total samples in dataset: {len(dataset)}")

    if args.num_samples > 0:
        dataset = dataset[: args.num_samples]
        logging.info(f"Evaluating first {args.num_samples} samples")

    # ── Load model ──
    model, processor = load_model(args.model_path)

    # ── Evaluate all samples ──
    all_results = []
    correct_samples = []
    incorrect_samples = []
    pattern_counts = {}

    for idx, item in enumerate(dataset):
        sample_id = item.get("id", idx)
        task_type = item.get("task_type", "unknown")
        logging.info(f"\n[{idx+1}/{len(dataset)}] id={sample_id}  task={task_type}")

        # Load video frames
        video_frames = {}
        for vkey in ["video_1", "video_2", "video_3", "video_4"]:
            vrel = item.get(vkey)
            if vrel is None:
                continue
            vpath = str(data_dir / vrel)
            if not os.path.exists(vpath):
                logging.warning(f"  Video not found: {vpath}")
                continue
            frames = extract_frames_from_video(vpath, args.num_frames)
            if frames:
                video_frames[vrel] = frames
                logging.info(f"  {vkey}: {len(frames)} frames loaded")

        if not video_frames:
            logging.warning("  No video frames loaded, skipping.")
            all_results.append({
                "id": sample_id, "task_type": task_type,
                "pred": "", "gt": item.get("answer", ""),
                "correct": False, "skipped": True,
            })
            continue

        # Build prompt with CoT
        messages, all_images, video_labels = build_prompt(
            item, video_frames, enable_cot=True
        )

        # Run inference
        try:
            full_text, clean_text = inference(
                model, processor, messages, all_images,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
            )
        except Exception as e:
            logging.error(f"  Inference error: {e}")
            all_results.append({
                "id": sample_id, "task_type": task_type,
                "pred": "", "gt": item.get("answer", ""),
                "correct": False, "error": str(e),
            })
            continue

        # Extract answer
        pred = extract_answer(clean_text, item.get("options", []))
        gt = item.get("answer", "")
        is_correct = check_correct(pred, gt)

        # Analyse reasoning pattern
        reasoning_analysis = analyse_reasoning_pattern(full_text, clean_text)
        pattern = reasoning_analysis["pattern"]
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # Error categorisation (only for wrong answers)
        error_category = ""
        if not is_correct:
            error_category = categorise_error(reasoning_analysis, item)

        logging.info(
            f"  Pred: {pred} | GT: {gt} | "
            f"{'CORRECT' if is_correct else 'WRONG'} | "
            f"Pattern: {pattern}"
        )

        result_entry = {
            "id": sample_id,
            "task_type": task_type,
            "question": item["question"],
            "options": item.get("options", []),
            "ground_truth": gt,
            "prediction": pred,
            "correct": is_correct,
            "full_output": full_text,
            "clean_output": clean_text,
            "reasoning_pattern": pattern,
            "thinking_text": reasoning_analysis["thinking_text"],
            "reasoning_text": reasoning_analysis["reasoning_text"],
            "answer_text": reasoning_analysis["answer_text"],
            "error_category": error_category,
            "videos_used": [
                vkey for vkey in ["video_1", "video_2", "video_3", "video_4"]
                if item.get(vkey) is not None
            ],
        }

        all_results.append(result_entry)
        if is_correct:
            correct_samples.append(result_entry)
        else:
            incorrect_samples.append(result_entry)

        # Periodic report
        total_so_far = len(correct_samples) + len(incorrect_samples)
        if (idx + 1) % 10 == 0:
            n_corr = len(correct_samples)
            logging.info(
                f"  ── Running: {n_corr}/{total_so_far} correct "
                f"({n_corr/total_so_far*100:.1f}%)"
            )

    # ════════════════════════════════════════════════════════════════════════
    # Select samples and build final analysis JSON
    # ════════════════════════════════════════════════════════════════════════
    N = args.num_select

    # Try to pick diverse task types
    def select_diverse(samples: list, n: int) -> list:
        """Select up to n samples, preferring diverse task_types."""
        if len(samples) <= n:
            return samples
        by_task = {}
        for s in samples:
            by_task.setdefault(s["task_type"], []).append(s)
        selected = []
        # Round-robin across task types
        tasks = list(by_task.keys())
        ti = 0
        while len(selected) < n:
            task = tasks[ti % len(tasks)]
            if by_task[task]:
                selected.append(by_task[task].pop(0))
            ti += 1
            # Safety: if all lists exhausted
            if all(len(v) == 0 for v in by_task.values()):
                break
        return selected[:n]

    selected_correct = select_diverse(correct_samples, N)
    selected_incorrect = select_diverse(incorrect_samples, N)

    # ── Overall statistics ──
    total_evaluated = len(correct_samples) + len(incorrect_samples)
    overall_accuracy = (
        len(correct_samples) / total_evaluated * 100
        if total_evaluated > 0 else 0
    )

    # ── Pattern analysis ──
    # For correct vs incorrect, compare pattern distributions
    def pattern_dist(samples):
        dist = {}
        for s in samples:
            p = s["reasoning_pattern"]
            dist[p] = dist.get(p, 0) + 1
        return dist

    correct_patterns = pattern_dist(correct_samples)
    incorrect_patterns = pattern_dist(incorrect_samples)

    # ── Build final output ──
    analysis_output = {
        "metadata": {
            "model": args.model_path.split("/")[-1],
            "dataset": "CVBench",
            "num_evaluated": total_evaluated,
            "num_correct": len(correct_samples),
            "num_incorrect": len(incorrect_samples),
            "overall_accuracy_pct": round(overall_accuracy, 2),
            "num_frames_per_video": args.num_frames,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": args.enable_thinking,
            "cot_prompt": True,
        },
        "reasoning_pattern_analysis": {
            "description": (
                "Reasoning patterns observed in model outputs. "
                "'reason-first' means the model describes observations and "
                "reasons before stating the answer. "
                "'answer-first' means the model states the answer letter "
                "immediately and then explains. "
                "'thinking-mode' means the model used <think>...</think> blocks. "
                "'answer-only' means the model gave a very short answer with "
                "minimal or no reasoning."
            ),
            "overall_pattern_counts": pattern_counts,
            "correct_samples_pattern_counts": correct_patterns,
            "incorrect_samples_pattern_counts": incorrect_patterns,
            "observation": "",  # filled below
        },
        "selected_correct_samples": [],
        "selected_incorrect_samples": [],
    }

    # Generate observation text
    obs_parts = []
    if correct_patterns and incorrect_patterns:
        # Most common pattern in correct vs incorrect
        most_common_correct = max(correct_patterns, key=correct_patterns.get)
        most_common_incorrect = max(incorrect_patterns, key=incorrect_patterns.get)
        obs_parts.append(
            f"Among correct answers, the most common pattern is "
            f"'{most_common_correct}' ({correct_patterns[most_common_correct]} times). "
            f"Among incorrect answers, the most common pattern is "
            f"'{most_common_incorrect}' ({incorrect_patterns[most_common_incorrect]} times)."
        )
        # Check if answer-first correlates with errors
        af_correct = correct_patterns.get("answer-first", 0)
        af_incorrect = incorrect_patterns.get("answer-first", 0)
        if af_incorrect > af_correct:
            obs_parts.append(
                "Answer-first pattern appears more frequently in incorrect "
                "answers, suggesting the model may commit to an answer "
                "prematurely without sufficient cross-video reasoning."
            )
        rf_correct = correct_patterns.get("reason-first", 0)
        rf_incorrect = incorrect_patterns.get("reason-first", 0)
        if rf_correct > rf_incorrect:
            obs_parts.append(
                "Reason-first pattern is more common in correct answers, "
                "indicating that step-by-step reasoning before answering "
                "may improve accuracy."
            )
    analysis_output["reasoning_pattern_analysis"]["observation"] = " ".join(obs_parts)

    # Add selected samples with annotations
    for s in selected_correct:
        entry = {
            "id": s["id"],
            "task_type": s["task_type"],
            "question": s["question"],
            "options": s["options"],
            "ground_truth": s["ground_truth"],
            "prediction": s["prediction"],
            "correct": True,
            "reasoning_pattern": s["reasoning_pattern"],
            "full_model_output": s["clean_output"],
            "thinking_block": s["thinking_text"],
            "reasoning_portion": s["reasoning_text"],
            "answer_portion": s["answer_text"],
            "videos_used": s["videos_used"],
            "annotation": (
                f"Pattern: {s['reasoning_pattern']}. "
                f"The model correctly answered {s['ground_truth']}."
            ),
        }
        analysis_output["selected_correct_samples"].append(entry)

    for s in selected_incorrect:
        entry = {
            "id": s["id"],
            "task_type": s["task_type"],
            "question": s["question"],
            "options": s["options"],
            "ground_truth": s["ground_truth"],
            "prediction": s["prediction"],
            "correct": False,
            "reasoning_pattern": s["reasoning_pattern"],
            "error_category": s["error_category"],
            "full_model_output": s["clean_output"],
            "thinking_block": s["thinking_text"],
            "reasoning_portion": s["reasoning_text"],
            "answer_portion": s["answer_text"],
            "videos_used": s["videos_used"],
            "annotation": (
                f"Pattern: {s['reasoning_pattern']}. "
                f"Error category: {s['error_category']}. "
                f"The model predicted {s['prediction']} but the correct "
                f"answer is {s['ground_truth']}."
            ),
        }
        analysis_output["selected_incorrect_samples"].append(entry)

    # ── Save outputs ──
    # 1. Full results (all samples)
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logging.info(f"All results saved to {all_results_path}")

    # 2. Analysis JSON (the main deliverable)
    analysis_path = output_dir / "reasoning_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis_output, f, indent=2, ensure_ascii=False)
    logging.info(f"Reasoning analysis saved to {analysis_path}")

    # ── Print summary to console ──
    print("\n" + "=" * 70)
    print("REASONING ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Samples evaluated : {total_evaluated}")
    print(f"Accuracy          : {overall_accuracy:.1f}%")
    print(f"Pattern counts    : {json.dumps(pattern_counts, indent=2)}")
    print(f"\nCorrect patterns  : {json.dumps(correct_patterns, indent=2)}")
    print(f"Incorrect patterns: {json.dumps(incorrect_patterns, indent=2)}")
    print(f"\nSelected {len(selected_correct)} correct + "
          f"{len(selected_incorrect)} incorrect samples → {analysis_path}")
    print("=" * 70)

    if obs_parts:
        print("\nKey observation:")
        print("  " + " ".join(obs_parts))


if __name__ == "__main__":
    main()