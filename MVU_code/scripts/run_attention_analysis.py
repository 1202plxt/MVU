#!/usr/bin/env python3
"""
多视频注意力可视化分析 — 单文件自包含脚本
==========================================
所有参数都在最顶部，直接改完 python run_attention_analysis.py 即可。
支持多卡分布 + INT4/INT8/BF16 + 逐层注意力提取到 CPU（不爆显存）。
"""

# ╔══════════════════════════════════════════════════════════════╗
# ║                    ★ 修改这里的参数 ★                        ║
# ╚══════════════════════════════════════════════════════════════╝

# -------------------- 路径 --------------------
PROJECT_ROOT     = "/fs0/AI/sx2624011/LKOPD/MVU"
MODEL_PATH       = f"{PROJECT_ROOT}/shared_models/Qwen3-VL-8B"
CVBENCH_JSON     = f"{PROJECT_ROOT}/MVU_data/datasets/CVBench/CVBench.json"
CVBENCH_VIDEO_DIR= f"{PROJECT_ROOT}/MVU_data/datasets/CVBench"
OUTPUT_DIR       = f"{PROJECT_ROOT}/MVU_data/outputs/attention_analysis"
VIS_DIR          = f"{PROJECT_ROOT}/MVU_data/visualizations/attention_maps"
LOG_DIR          = f"{PROJECT_ROOT}/MVU_data/logs"
HF_CACHE         = f"{PROJECT_ROOT}/MVU_data/cache"

# -------------------- GPU & 量化 --------------------
GPU_IDS          = [0, 1]          # 用哪些卡, 单卡就写 [0]
QUANTIZATION     = "int4"          # "int4" | "int8" | "bf16"
                                   #   int4: ~5GB, 1卡够
                                   #   int8: ~8GB
                                   #   bf16: ~16GB, 建议2卡

# -------------------- 视频采样 --------------------
FRAMES_PER_VIDEO = 8               # 每视频采样帧数 (4/8/16)
MAX_VIDEOS       = 4               # 每样本最多用几个视频
MAX_PIXELS       = 360000          # 每帧最大像素数 (~600x600)
MIN_PIXELS       = 100000          # 每帧最小像素数 (~316x316)

# -------------------- 注意力提取 --------------------
# Qwen3-VL-8B 共 28 层 (0~27)
# "selected" = 只提取指定层,  "all" = 全部 28 层
EXTRACT_MODE     = "selected"
SELECTED_LAYERS  = [0, 2, 5, 8, 12, 14, 16, 20, 24, 27]
SAVE_RAW_ATTN    = False           # True 会存很大的 .npz 文件
SAVE_FORMAT      = "npz"           # "npz" | "pt"

# -------------------- 样本选择 --------------------
SAMPLE_IDS       = []              # 指定样本ID列表, 空=自动选
NUM_SAMPLES      = 20              # 自动选取数量
ONLY_WRONG       = True            # 只分析答错的样本
INCLUDE_CORRECT  = True            # 同时抓一些答对的做对照

# -------------------- Activation Patching --------------------
DO_PATCHING      = True            # 是否做 activation patching
PATCHING_NOISE   = "zero"          # "zero" | "gaussian" | "shuffle"

# -------------------- 可视化 --------------------
GENERATE_HEATMAPS     = True
GENERATE_FRAME_OVERLAY= True
VIS_DPI               = 150
VIS_CMAP              = "RdYlBu_r"

# -------------------- 其他 --------------------
SEED             = 42
LOG_LEVEL        = "INFO"          # "DEBUG" | "INFO" | "WARNING"
ENABLE_THINKING  = False           # Qwen3-VL thinking 模式

# ╔══════════════════════════════════════════════════════════════╗
# ║                  ★ 以下是代码，一般不用改 ★                   ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import sys
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from PIL import Image

# ============================================================
#  环境设置
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPU_IDS)
os.environ["HF_HOME"] = HF_CACHE
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

for d in [OUTPUT_DIR, VIS_DIR, LOG_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/attn_{timestamp}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
#  1. 模型加载 (多卡 + 量化)
# ============================================================

def load_model():
    """加载 Qwen3-VL-8B, 支持 INT4/INT8/BF16 + 多卡 auto 分配"""
    from transformers import AutoProcessor

    logging.info(f"加载模型: {MODEL_PATH}")
    logging.info(f"量化: {QUANTIZATION} | GPU: {GPU_IDS}")

    load_kwargs = dict(
        pretrained_model_name_or_path=MODEL_PATH,
        attn_implementation="eager",     # 必须 eager 才返回注意力权重
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",               # accelerate 自动分配多卡
    )

    if QUANTIZATION == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif QUANTIZATION == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif QUANTIZATION == "bf16":
        if len(GPU_IDS) == 1:
            load_kwargs["device_map"] = {"": "cuda:0"}
    else:
        raise ValueError(f"不支持的量化: {QUANTIZATION}")

    # 尝试 Qwen2.5VL 类名 (transformers 版本兼容)
    model = None
    for cls_name in [
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "AutoModelForCausalLM",
    ]:
        try:
            from transformers import AutoModelForCausalLM
            cls = getattr(__import__("transformers", fromlist=[cls_name]), cls_name, None)
            if cls is None:
                continue
            model = cls.from_pretrained(**load_kwargs)
            logging.info(f"模型类: {cls_name}")
            break
        except Exception as e:
            logging.debug(f"{cls_name} 失败: {e}")
            continue

    if model is None:
        raise RuntimeError("无法加载模型，请检查 MODEL_PATH 和 transformers 版本")

    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / 1024**3
        logging.info(f"GPU {i} 显存: {mem:.2f} GB")

    return model, processor


# ============================================================
#  2. 数据加载 (CVBench)
# ============================================================

def load_cvbench_samples() -> List[dict]:
    """加载并筛选 CVBench 样本"""
    logging.info(f"加载 CVBench: {CVBENCH_JSON}")
    with open(CVBENCH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 兼容 list / dict 格式
    if isinstance(data, list):
        samples = data
        for i, s in enumerate(samples):
            s.setdefault("sample_id", str(i))
    elif isinstance(data, dict):
        samples = []
        for k, v in data.items():
            if isinstance(v, dict):
                v["sample_id"] = str(k)
                samples.append(v)
    else:
        raise ValueError("无法解析 CVBench.json")

    # 挂载视频路径
    for s in samples:
        sid = str(s["sample_id"])
        vid_folder = os.path.join(CVBENCH_VIDEO_DIR, sid)
        if os.path.isdir(vid_folder):
            s["video_paths"] = sorted([
                os.path.join(vid_folder, f)
                for f in os.listdir(vid_folder)
                if f.endswith((".mp4", ".avi", ".mkv", ".mov"))
            ])
        else:
            s["video_paths"] = []

    # 只保留有 ≥2 个视频的样本
    samples = [s for s in samples if len(s.get("video_paths", [])) >= 2]
    logging.info(f"有效样本(≥2 视频): {len(samples)}")

    # 指定样本 or 自动选取
    if SAMPLE_IDS:
        id_set = set(str(i) for i in SAMPLE_IDS)
        samples = [s for s in samples if s["sample_id"] in id_set]
    else:
        samples = samples[:NUM_SAMPLES]

    logging.info(f"本次分析: {len(samples)} 个样本")
    return samples


# ============================================================
#  3. 视频帧采样
# ============================================================

def sample_frames(video_path: str, n_frames: int) -> List[Image.Image]:
    """均匀采样视频帧, 优先 decord, fallback opencv"""
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        if total == 0:
            return []
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f) for f in frames]
    except ImportError:
        pass

    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


# ============================================================
#  4. 注意力提取器 (逐层 Hook → CPU, 不爆显存)
# ============================================================

class AttentionExtractor:
    """
    核心: 通过 forward hook 逐层提取注意力矩阵, 立即 .cpu()。
    GPU 上永远只有当前层的注意力, 之前层的已经在 CPU 内存里了。
    """

    def __init__(self, model):
        self.model = model
        self.attention_maps: Dict[int, torch.Tensor] = {}
        self.hooks = []

        layers = SELECTED_LAYERS if EXTRACT_MODE == "selected" else list(range(28))
        logging.info(f"注册 hook: 层 {layers}")

        for idx in layers:
            try:
                # Qwen3-VL 层结构: model.model.layers[i].self_attn
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    module = model.model.layers[idx].self_attn
                elif hasattr(model, "layers"):
                    module = model.layers[idx].self_attn
                else:
                    logging.warning(f"无法定位层 {idx}")
                    continue

                hook = module.register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)
            except (IndexError, AttributeError) as e:
                logging.warning(f"层 {idx} hook 失败: {e}")

    def _make_hook(self, layer_idx: int):
        """hook 函数: 拿到注意力立即转 CPU"""
        def fn(module, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                # out[1] shape: [batch, num_heads, seq_len, seq_len]
                self.attention_maps[layer_idx] = out[1].detach().cpu().float()
        return fn

    def clear(self):
        self.attention_maps.clear()
        torch.cuda.empty_cache()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ============================================================
#  5. 构建输入 + Token 归属映射
# ============================================================

def build_input_and_token_map(
    processor,
    video_frames: List[List[Image.Image]],
    question: str,
    options: str,
) -> Tuple[dict, dict]:
    """
    把多视频帧 + 问题组装成模型输入, 同时记录
    每段 token 归属于 video_0 / video_1 / ... / question。
    """
    # 组装 Qwen3-VL 多图消息
    content = []
    for vid_idx, frames in enumerate(video_frames):
        for frame in frames:
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": f"\n[Video {vid_idx + 1} 以上]\n"})

    full_q = f"{question}\n{options}" if options else question
    content.append({"type": "text", "text": full_q})

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[f for frames in video_frames for f in frames],
        padding=True,
        return_tensors="pt",
    )

    # ---- 构建 token 归属 ----
    input_ids = inputs["input_ids"][0]
    total = len(input_ids)
    token_map = {"total_tokens": total, "frames_per_video": {}}

    # 找 vision pad token 位置
    vision_token_id = getattr(processor, "image_token_id", None)
    if vision_token_id is None:
        try:
            vision_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            vision_token_id = 151655   # Qwen3-VL 默认值

    vision_pos = (input_ids == vision_token_id).nonzero(as_tuple=True)[0].tolist()

    if "image_grid_thw" in inputs and vision_pos:
        # 精确分配: 每张图实际占多少 token
        grids = inputs["image_grid_thw"]
        tokens_per_img = [int(g[0] * g[1] * g[2]) for g in grids]

        cursor = 0
        img_idx = 0
        for vid_idx, frames in enumerate(video_frames):
            vid_start = vid_end = None
            for _ in frames:
                if img_idx < len(tokens_per_img) and cursor < len(vision_pos):
                    n = tokens_per_img[img_idx]
                    if vid_start is None:
                        vid_start = vision_pos[cursor]
                    end_c = min(cursor + n - 1, len(vision_pos) - 1)
                    vid_end = vision_pos[end_c]
                    cursor += n
                    img_idx += 1

            if vid_start is not None:
                token_map[f"video_{vid_idx}"] = (vid_start, vid_end + 1)
                token_map["frames_per_video"][vid_idx] = len(frames)
    else:
        # fallback: 均等切分
        n_vision = len(vision_pos)
        n_vids = len(video_frames)
        per = n_vision // n_vids if n_vids else 0
        for vid_idx in range(n_vids):
            s, e = vid_idx * per, (vid_idx + 1) * per
            if s < len(vision_pos) and e <= len(vision_pos):
                token_map[f"video_{vid_idx}"] = (vision_pos[s], vision_pos[e - 1] + 1)

    # 问题 token 范围
    all_vid_end = max(
        (v[1] for k, v in token_map.items() if k.startswith("video_") and isinstance(v, tuple)),
        default=0,
    )
    token_map["question"] = (all_vid_end, total)

    # 打印映射
    for k, v in token_map.items():
        if isinstance(v, tuple):
            logging.info(f"  token_map  {k}: [{v[0]}, {v[1]}) = {v[1]-v[0]} tokens")

    return inputs, token_map


# ============================================================
#  6. 注意力统计分析
# ============================================================

def analyze_attention(
    attn_maps: Dict[int, torch.Tensor],
    token_map: dict,
    answer_idx: int,
) -> dict:
    """
    统计:
      1) 答案 token 对各视频/问题区域的注意力分配
      2) 每层的跨视频 vs 同视频注意力比例
    """
    video_ranges = {
        k: v for k, v in token_map.items()
        if k.startswith("video_") and isinstance(v, tuple)
    }
    q_range = token_map.get("question", (0, 0))

    stats = {"answer_token_idx": answer_idx, "per_layer": {}}

    for layer_idx, attn in attn_maps.items():
        attn = attn[0]  # [heads, seq, seq]
        heads, seq_len, _ = attn.shape
        ls = {"num_heads": heads, "seq_len": seq_len}

        # ---- 答案 token → 各区域 ----
        if answer_idx < seq_len:
            ans_row = attn[:, answer_idx, :]     # [heads, seq]
            region_attn = {}
            for name, (s, e) in video_ranges.items():
                e = min(e, seq_len)
                if s < e:
                    vals = ans_row[:, s:e].sum(dim=-1)  # [heads]
                    region_attn[name] = {
                        "mean": vals.mean().item(),
                        "per_head": vals.tolist(),
                    }
            qs, qe = q_range
            qe = min(qe, seq_len)
            if qs < qe:
                vals = ans_row[:, qs:qe].sum(dim=-1)
                region_attn["question"] = {
                    "mean": vals.mean().item(),
                    "per_head": vals.tolist(),
                }
            ls["answer_to_regions"] = region_attn

        # ---- 跨视频 vs 同视频 ----
        cross, same, total_v = 0.0, 0.0, 0.0
        for n1, (s1, e1) in video_ranges.items():
            e1 = min(e1, seq_len)
            for n2, (s2, e2) in video_ranges.items():
                e2 = min(e2, seq_len)
                if s1 >= e1 or s2 >= e2:
                    continue
                block = attn[:, s1:e1, s2:e2].sum().item()
                total_v += block
                if n1 == n2:
                    same += block
                else:
                    cross += block
        ls["cross_video_ratio"] = cross / total_v if total_v > 0 else 0.0
        ls["same_video_ratio"]  = same  / total_v if total_v > 0 else 0.0

        stats["per_layer"][layer_idx] = ls

    return stats


# ============================================================
#  7. Activation Patching
# ============================================================

def run_patching(model, inputs: dict, token_map: dict) -> dict:
    """抹掉某个视频的 hidden state, 观察输出 logit 变化"""
    if not DO_PATCHING:
        return {}

    video_ranges = {
        k: v for k, v in token_map.items()
        if k.startswith("video_") and isinstance(v, tuple)
    }

    # baseline logits
    device = next(model.parameters()).device
    inp = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.no_grad():
        base_logits = model(**inp, output_attentions=False).logits[0, -1, :].cpu().float()

    results = {}
    for vname, (vs, ve) in video_ranges.items():

        def make_hook(vs, ve):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if PATCHING_NOISE == "zero":
                    h[:, vs:ve, :] = 0.0
                elif PATCHING_NOISE == "gaussian":
                    h[:, vs:ve, :] = torch.randn_like(h[:, vs:ve, :])
                elif PATCHING_NOISE == "shuffle":
                    idx = torch.randperm(ve - vs)
                    h[:, vs:ve, :] = h[:, vs:ve, :][:, idx, :]
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return fn

        # hook embedding 层
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            target = model.model.embed_tokens
        else:
            continue

        hook = target.register_forward_hook(make_hook(vs, ve))
        with torch.no_grad():
            patched_logits = model(**inp, output_attentions=False).logits[0, -1, :].cpu().float()
        hook.remove()

        diff = base_logits - patched_logits
        results[vname] = {
            "logit_diff_norm": diff.norm().item(),
            "logit_diff_max":  diff.abs().max().item(),
            "top_affected":    diff.abs().topk(10).indices.tolist(),
        }
        logging.info(f"  patching {vname}: diff_norm={diff.norm():.2f}, diff_max={diff.abs().max():.2f}")

    return results


# ============================================================
#  8. 可视化
# ============================================================

def generate_visualizations(all_stats: List[dict]):
    """生成注意力分析图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    viz_dir = Path(VIS_DIR)

    for ss in all_stats:
        sid = ss["sample_id"]
        is_correct = ss.get("is_correct")
        per_layer = ss.get("attention_stats", {}).get("per_layer", {})
        if not per_layer:
            continue

        layers_sorted = sorted(per_layer.keys())
        status = "correct" if is_correct else "wrong" if is_correct is not None else "unknown"
        status_cn = "✓答对" if is_correct else "✗答错" if is_correct is not None else ""

        # ---- 图1: 答案 token → 各区域注意力 ----
        fig, ax = plt.subplots(figsize=(14, 6))
        region_names = set()
        for ld in per_layer.values():
            region_names.update(ld.get("answer_to_regions", {}).keys())
        region_names = sorted(region_names)

        for rname in region_names:
            vals = [per_layer[l].get("answer_to_regions", {}).get(rname, {}).get("mean", 0)
                    for l in layers_sorted]
            label = rname.replace("video_", "Video ").replace("question", "Question")
            ax.plot(layers_sorted, vals, marker="o", label=label, linewidth=2)

        ax.set_title(f"样本 {sid} {status_cn} — 答案Token→各区域注意力", fontsize=14)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention Sum")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(viz_dir / f"{sid}_{status}_answer_attn.png", dpi=VIS_DPI)
        plt.close(fig)

        # ---- 图2: 跨视频 vs 同视频注意力比 ----
        fig, ax = plt.subplots(figsize=(14, 5))
        same_r  = [per_layer[l].get("same_video_ratio", 0) for l in layers_sorted]
        cross_r = [per_layer[l].get("cross_video_ratio", 0) for l in layers_sorted]

        ax.bar(layers_sorted, same_r,  label="同视频", alpha=0.7, color="steelblue")
        ax.bar(layers_sorted, cross_r, bottom=same_r, label="跨视频", alpha=0.7, color="coral")
        ax.set_title(f"样本 {sid} {status_cn} — 跨/同视频注意力比", fontsize=14)
        ax.set_xlabel("Layer")
        ax.set_ylabel("比例")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(viz_dir / f"{sid}_{status}_cross_ratio.png", dpi=VIS_DPI)
        plt.close(fig)

    # ---- 汇总: 答对 vs 答错 ----
    correct_vals = []
    wrong_vals = []
    for ss in all_stats:
        pl = ss.get("attention_stats", {}).get("per_layer", {})
        if not pl or ss.get("is_correct") is None:
            continue
        avg = np.mean([v.get("cross_video_ratio", 0) for v in pl.values()])
        if ss["is_correct"]:
            correct_vals.append(avg)
        else:
            wrong_vals.append(avg)

    if correct_vals or wrong_vals:
        fig, ax = plt.subplots(figsize=(8, 6))
        data, labels = [], []
        if correct_vals:
            data.append(correct_vals); labels.append(f"答对 (n={len(correct_vals)})")
        if wrong_vals:
            data.append(wrong_vals);   labels.append(f"答错 (n={len(wrong_vals)})")

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#2ecc71", "#e74c3c"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)

        ax.set_title("答对 vs 答错: 平均跨视频注意力比例", fontsize=14)
        ax.set_ylabel("Cross-Video Attention Ratio")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(viz_dir / "summary_correct_vs_wrong.png", dpi=VIS_DPI)
        plt.close(fig)

    logging.info(f"可视化已保存: {viz_dir}")


# ============================================================
#  9. 单样本处理
# ============================================================

def process_sample(model, processor, extractor: AttentionExtractor, sample: dict) -> dict:
    """处理一个 CVBench 样本: 推理 → 提取注意力 → 分析 → 返回统计"""
    sid = sample["sample_id"]
    video_paths = sample["video_paths"]
    question = sample.get("question", sample.get("Q", ""))
    options  = sample.get("options",  sample.get("choices", ""))
    if isinstance(options, list):
        options = "\n".join(options)
    gt = sample.get("answer", sample.get("A", ""))

    logging.info(f"\n{'='*60}")
    logging.info(f"样本 {sid} | {len(video_paths)} 个视频 | 问题: {question[:80]}...")

    # 采样帧
    all_frames = []
    for vp in video_paths[:MAX_VIDEOS]:
        frames = sample_frames(vp, FRAMES_PER_VIDEO)
        if frames:
            all_frames.append(frames)
            logging.info(f"  {os.path.basename(vp)}: {len(frames)} 帧")
    if len(all_frames) < 2:
        return {"sample_id": sid, "skipped": True, "reason": "帧不足"}

    # 构建输入
    inputs, token_map = build_input_and_token_map(processor, all_frames, question, options)

    # forward + 注意力提取
    extractor.clear()
    device = next(model.parameters()).device
    inp = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    logging.info("  forward (output_attentions=True)...")
    with torch.no_grad():
        outputs = model.generate(
            **inp,
            max_new_tokens=32,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

    # 解码预测
    gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    pred = processor.decode(gen_ids, skip_special_tokens=True).strip()
    logging.info(f"  预测: {pred} | 正确: {gt}")

    is_correct = None
    if gt:
        is_correct = str(gt).strip().upper()[0] == pred.strip().upper()[0] if pred else None

    # 注意力分析
    answer_idx = inputs["input_ids"].shape[1] - 1
    attn_stats = analyze_attention(extractor.attention_maps, token_map, answer_idx)

    # activation patching
    patch_res = run_patching(model, inputs, token_map)

    # 保存原始注意力 (可选)
    if SAVE_RAW_ATTN:
        raw_dir = Path(OUTPUT_DIR) / f"raw_attn/sample_{sid}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for li, at in extractor.attention_maps.items():
            if SAVE_FORMAT == "npz":
                np.savez_compressed(raw_dir / f"layer_{li}.npz", attention=at.numpy())
            else:
                torch.save(at, raw_dir / f"layer_{li}.pt")

    extractor.clear()
    del inp
    torch.cuda.empty_cache()

    # 打印关键统计
    tag = "✓" if is_correct else "✗" if is_correct is not None else "?"
    logging.info(f"  结果: {tag}")
    for li in sorted(attn_stats["per_layer"]):
        ls = attn_stats["per_layer"][li]
        cr = ls.get("cross_video_ratio", 0)
        parts = " | ".join(
            f"{k}:{v['mean']:.4f}"
            for k, v in ls.get("answer_to_regions", {}).items()
        )
        logging.info(f"  L{li:2d}: cross={cr:.4f} | ans→ {parts}")

    return {
        "sample_id": sid,
        "question": question,
        "ground_truth": gt,
        "prediction": pred,
        "is_correct": is_correct,
        "token_map": {k: list(v) if isinstance(v, tuple) else v
                      for k, v in token_map.items()},
        "attention_stats": attn_stats,
        "patching_results": patch_res,
    }


# ============================================================
#  10. 主函数
# ============================================================

def main():
    logging.info("=" * 60)
    logging.info("多视频注意力可视化分析")
    logging.info(f"GPU: {GPU_IDS} | 量化: {QUANTIZATION} | 帧/视频: {FRAMES_PER_VIDEO}")
    logging.info(f"提取层: {SELECTED_LAYERS if EXTRACT_MODE=='selected' else 'ALL 28'}")
    logging.info("=" * 60)

    model, processor = load_model()
    extractor = AttentionExtractor(model)
    samples = load_cvbench_samples()

    all_stats = []
    for i, s in enumerate(samples):
        logging.info(f"\n进度: {i+1}/{len(samples)}")
        try:
            result = process_sample(model, processor, extractor, s)
            all_stats.append(result)

            # 每个样本跑完就中间保存 (防中断丢失)
            with open(f"{OUTPUT_DIR}/attn_stats_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logging.error(f"样本 {s.get('sample_id','?')} 失败: {e}")
            traceback.print_exc()

    # 可视化
    if GENERATE_HEATMAPS and all_stats:
        generate_visualizations(all_stats)

    # 最终保存
    final_path = f"{OUTPUT_DIR}/attn_stats_final.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2, default=str)

    correct = sum(1 for s in all_stats if s.get("is_correct") is True)
    wrong   = sum(1 for s in all_stats if s.get("is_correct") is False)
    logging.info(f"\n{'='*60}")
    logging.info(f"完成! 共 {len(all_stats)} 样本, 答对 {correct}, 答错 {wrong}")
    logging.info(f"结果: {final_path}")
    logging.info(f"图表: {VIS_DIR}")

    extractor.remove_hooks()


if __name__ == "__main__":
    main()
