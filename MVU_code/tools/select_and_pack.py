#!/usr/bin/env python3
"""
从 CVBench 评测结果中选择 10 个正确样本和 10 个错误样本，
打包对应的视频文件为 zip 压缩包。

用法：
    python select_and_pack.py \
        --results_json /path/to/all_results.json \
        --cvbench_json /path/to/CVBench.json \
        --video_dir /path/to/CVBench/ \
        --output_zip /path/to/output.zip \
        --num_samples 10

输出结构：
    selected_samples/
    ├── correct/
    │   ├── sample_0/
    │   │   ├── info.json          # 问题、GT、预测、任务类型等元信息
    │   │   ├── video1.mp4
    │   │   └── video2.mp4
    │   ├── sample_3/
    │   │   └── ...
    │   └── ...
    └── incorrect/
        ├── sample_1/
        │   ├── info.json
        │   ├── video1.mp4
        │   └── video2.mp4
        └── ...
"""

import json
import os
import shutil
import argparse
import zipfile
from pathlib import Path


def load_results(results_path):
    """加载评测结果 JSON"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_cvbench(cvbench_path):
    """加载 CVBench.json 标注文件，建立 sample_id -> 标注信息 的映射"""
    with open(cvbench_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 尝试理解数据结构
    if isinstance(data, list):
        # 如果是列表，尝试用索引或 id 字段建立映射
        mapping = {}
        for i, item in enumerate(data):
            if 'id' in item:
                mapping[item['id']] = item
            elif 'sample_id' in item:
                mapping[item['sample_id']] = item
            mapping[i] = item  # 也用索引映射
        return data, mapping
    elif isinstance(data, dict):
        return data, data
    
    return data, {}


def find_videos_for_sample(video_dir, sample_id):
    """找到某个样本对应的视频文件"""
    # CVBench 结构: video_dir/sample_id/video1.mp4, video2.mp4, ...
    sample_folder = os.path.join(video_dir, str(sample_id))
    
    if not os.path.isdir(sample_folder):
        print(f"  警告: 文件夹不存在 {sample_folder}")
        return []
    
    videos = []
    for f in sorted(os.listdir(sample_folder)):
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
            videos.append(os.path.join(sample_folder, f))
    
    return videos


def classify_samples(results):
    """将样本分为正确和错误两组"""
    correct = []
    incorrect = []
    
    # 适配不同的结果格式
    if isinstance(results, list):
        for item in results:
            sid = item.get('sample_id', item.get('id', item.get('index')))
            pred = item.get('prediction', item.get('pred', item.get('direct_pred')))
            gt = item.get('ground_truth', item.get('gt', item.get('answer', item.get('GT'))))
            
            is_correct = item.get('correct', item.get('direct_result', None))
            
            # 如果没有 correct 字段，自行比较
            if is_correct is None and pred is not None and gt is not None:
                is_correct = str(pred).strip().upper() == str(gt).strip().upper()
            
            record = {
                'sample_id': sid,
                'prediction': pred,
                'ground_truth': gt,
                'task_type': item.get('task_type', item.get('type', 'unknown')),
                'question': item.get('question', ''),
                'options': item.get('options', item.get('choices', '')),
                'raw': item  # 保留原始数据
            }
            
            if is_correct:
                correct.append(record)
            else:
                incorrect.append(record)
    
    elif isinstance(results, dict):
        for sid, item in results.items():
            if isinstance(item, dict):
                pred = item.get('prediction', item.get('pred', item.get('direct_pred')))
                gt = item.get('ground_truth', item.get('gt', item.get('answer', item.get('GT'))))
                
                is_correct = item.get('correct', item.get('direct_result', None))
                if is_correct is None and pred is not None and gt is not None:
                    is_correct = str(pred).strip().upper() == str(gt).strip().upper()
                
                record = {
                    'sample_id': sid,
                    'prediction': pred,
                    'ground_truth': gt,
                    'task_type': item.get('task_type', item.get('type', 'unknown')),
                    'question': item.get('question', ''),
                    'options': item.get('options', item.get('choices', '')),
                    'raw': item
                }
                
                if is_correct:
                    correct.append(record)
                else:
                    incorrect.append(record)
    
    return correct, incorrect


def select_diverse_samples(samples, n=10):
    """尽量选择不同任务类型的样本，保证多样性"""
    if len(samples) <= n:
        return samples
    
    # 按任务类型分组
    by_type = {}
    for s in samples:
        t = s.get('task_type', 'unknown')
        by_type.setdefault(t, []).append(s)
    
    selected = []
    types = list(by_type.keys())
    
    # 轮询每个类型
    idx = {t: 0 for t in types}
    while len(selected) < n:
        for t in types:
            if len(selected) >= n:
                break
            if idx[t] < len(by_type[t]):
                selected.append(by_type[t][idx[t]])
                idx[t] += 1
    
    return selected[:n]


def pack_samples(correct, incorrect, video_dir, output_zip, staging_dir='/tmp/mvbench_selected'):
    """打包选中样本的视频到 zip"""
    # 清理暂存目录
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir)
    
    summary = {'correct': [], 'incorrect': []}
    
    for label, samples in [('correct', correct), ('incorrect', incorrect)]:
        for s in samples:
            sid = s['sample_id']
            sample_dir = os.path.join(staging_dir, label, f'sample_{sid}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 复制视频
            videos = find_videos_for_sample(video_dir, sid)
            for v in videos:
                dst = os.path.join(sample_dir, os.path.basename(v))
                shutil.copy2(v, dst)
            
            # 写入元信息
            info = {
                'sample_id': sid,
                'task_type': s['task_type'],
                'question': s['question'],
                'options': s['options'],
                'ground_truth': s['ground_truth'],
                'prediction': s['prediction'],
                'result': label,
                'num_videos': len(videos),
                'video_files': [os.path.basename(v) for v in videos]
            }
            
            with open(os.path.join(sample_dir, 'info.json'), 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            summary[label].append({
                'sample_id': sid,
                'task_type': s['task_type'],
                'num_videos': len(videos)
            })
            
            print(f"  [{label}] Sample {sid} ({s['task_type']}): {len(videos)} videos")
    
    # 写入总结文件
    with open(os.path.join(staging_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打包 zip
    print(f"\n正在创建压缩包 {output_zip} ...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(staging_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, staging_dir)
                zf.write(filepath, arcname)
    
    zip_size = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"完成! 压缩包大小: {zip_size:.1f} MB")
    
    # 清理暂存
    shutil.rmtree(staging_dir)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='选择 CVBench 正确/错误样本并打包视频')
    parser.add_argument('--results_json', required=True, help='评测结果 JSON 路径')
    parser.add_argument('--video_dir', required=True, help='CVBench 视频数据集根目录 (含 0/~467/ 文件夹)')
    parser.add_argument('--output_zip', default='selected_cvbench_samples.zip', help='输出 zip 路径')
    parser.add_argument('--num_samples', type=int, default=10, help='每组选择的样本数')
    args = parser.parse_args()
    
    # 1. 加载结果
    print("=" * 60)
    print("CVBench 样本选择与打包工具")
    print("=" * 60)
    
    print(f"\n1. 加载评测结果: {args.results_json}")
    results = load_results(args.results_json)
    
    if isinstance(results, list):
        print(f"   共 {len(results)} 个样本")
    elif isinstance(results, dict):
        print(f"   共 {len(results)} 个条目")
    
    # 先打印前 2 个样本的结构以便调试
    print("\n   前 2 个样本的 keys:")
    if isinstance(results, list):
        for i, item in enumerate(results[:2]):
            print(f"   [{i}] keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
    elif isinstance(results, dict):
        for i, (k, v) in enumerate(list(results.items())[:2]):
            print(f"   [{k}] keys: {list(v.keys()) if isinstance(v, dict) else type(v)}")
    
    # 2. 分类
    print(f"\n2. 分类正确/错误样本...")
    correct, incorrect = classify_samples(results)
    print(f"   正确: {len(correct)} 个")
    print(f"   错误: {len(incorrect)} 个")
    
    if not correct:
        print("   警告: 没有找到正确样本！检查结果 JSON 格式")
    if not incorrect:
        print("   警告: 没有找到错误样本！检查结果 JSON 格式")
    
    # 3. 选择多样化样本
    n = args.num_samples
    print(f"\n3. 选择每组 {n} 个样本（尽量覆盖不同任务类型）...")
    selected_correct = select_diverse_samples(correct, n)
    selected_incorrect = select_diverse_samples(incorrect, n)
    
    print(f"   选中正确样本: {len(selected_correct)} 个")
    print(f"   选中错误样本: {len(selected_incorrect)} 个")
    
    # 4. 打包
    print(f"\n4. 复制视频并打包...")
    summary = pack_samples(
        selected_correct, selected_incorrect,
        args.video_dir, args.output_zip
    )
    
    # 5. 最终总结
    print(f"\n{'=' * 60}")
    print("完成总结:")
    print(f"  正确样本: {len(summary['correct'])} 个")
    for s in summary['correct']:
        print(f"    - Sample {s['sample_id']} ({s['task_type']}): {s['num_videos']} videos")
    print(f"  错误样本: {len(summary['incorrect'])} 个")
    for s in summary['incorrect']:
        print(f"    - Sample {s['sample_id']} ({s['task_type']}): {s['num_videos']} videos")
    print(f"\n  输出文件: {args.output_zip}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()