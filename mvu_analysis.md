# Multi-Video Understanding (MVU) Analysis

## Current Codebase Analysis
The current codebase focuses on evaluating and interpreting how Vision-Language Models (VLMs), specifically Qwen3-VL-8B, process and reason over multiple videos simultaneously. It targets the CVBench dataset. The core functionalities include:

1. **Reasoning Pattern Analysis (`eval_cvbench_reasoning.py`)**:
   - Evaluates the model on CVBench using Chain-of-Thought (CoT) prompting or native thinking mode.
   - Categorizes reasoning patterns (e.g., "reason-first", "answer-first", "thinking-mode") to understand if the model genuinely reasons before answering or merely hallucinates a justification post-hoc.
   - Extracts correct and incorrect samples for deeper study.

2. **Perception vs. Reasoning Decoupling (`diagnose_perception_vs_reasoning.py`)**:
   - A multi-stage diagnostic approach to isolate whether errors stem from visual perception failures or reasoning logic flaws.
   - Stage 1: Asks the model to describe each video independently (Perception Probe).
   - Stage 2: Feeds the model's *own* descriptions back as text-only prompts to answer the question (Text-Only Reasoning).
   - Diagnoses errors by comparing direct video QA accuracy with text-only QA accuracy.

3. **Mechanistic Interpretability and Attention Analysis (`run_attention_analysis.py`, `visualize_attention.py`)**:
   - Uses PyTorch forward hooks to extract attention maps layer-by-layer during inference without causing OOM.
   - Analyzes how output tokens attend to different video frames (cross-video vs. same-video attention ratios).
   - Implements Activation Patching (zeroing, Gaussian noise, or shuffling) to measure the causal impact of specific video hidden states on the final prediction.
   - Generates detailed heatmaps to visualize the model's internal focus.

4. **Utility Scripts (`select_and_pack.py`)**:
   - Facilitates the curation of correct and incorrect samples, packaging videos and metadata into a zip for easy sharing and manual review.

## Proposed Next Steps for Multi-Video Understanding Analysis

Based on the existing tooling and analysis scripts, here are promising directions for advancing the study of multi-video understanding:

1. **Expanding VLM Compatibility and Dataset Breadth**:
   - The current pipeline is heavily optimized for Qwen3-VL-8B. The `AttentionExtractor` and prompt templates should be generalized to support other open-source multi-modal models (e.g., LLaVA, InternVL, Video-LLaMA).
   - Extend evaluation beyond CVBench to other multi-video benchmarks to verify if the observed reasoning patterns and perception bottlenecks generalize across different domains.

2. **Deepening Mechanistic Interpretability**:
   - **Fine-grained Activation Patching**: Move beyond coarse masking of entire videos to patching specific frames or bounding boxes within frames to isolate exactly which visual elements drive the final decision.
   - **Attention Head Identification**: Investigate specific attention heads (e.g., "comparison heads") across layers that consistently facilitate cross-video information synthesis.

3. **Developing Intervention Strategies**:
   - Utilize the insights from the "Perception vs. Reasoning" diagnostic to design interventions. If perception is the bottleneck, experiment with dynamically allocating higher resolution to frames that models struggle to describe.
   - If reasoning is the bottleneck (e.g., the model exhibits "answer-first" bias), design training or prompt engineering strategies (e.g., forced structured reasoning templates or multi-agent debate) to correct premature commitments.
   - Implement attention masking or guidance during inference to force the model to distribute attention more evenly across videos if "layer evenness" metrics drop.
