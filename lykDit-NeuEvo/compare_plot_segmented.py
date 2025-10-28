#!/usr/bin/env python3
"""
分段对比版：从两个训练日志中提取真实valid acc并按epoch段绘制对比曲线
分段：0-50, 51-100, 101-150
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def extract_valid_acc_from_log(log_file_path):
    """
    从单个日志文件中提取当前epoch的真实valid acc数据
    返回 (epochs, valid_accs) 元组
    """
    # 读取日志文件
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(log_file_path, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            with open(log_file_path, 'r', encoding='latin-1') as f:
                content = f.read()
    
    # 提取所有epoch和对应的真实valid acc
    epochs = []
    valid_accs = []
    
    lines = content.split('\n')
    current_epoch = 0
    
    for line in lines:
        # 查找epoch信息
        epoch_match = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 查找真实valid acc信息（不包括best valid acc）
        # 使用负向后顾断言，确保匹配的是 "valid acc" 而不是 "best valid acc"
        acc_match = re.search(r'(?<!best )valid acc ([\d.]+)', line)
        if acc_match:
            valid_acc = float(acc_match.group(1))
            epochs.append(current_epoch)
            valid_accs.append(valid_acc)
            print(f"文件 {os.path.basename(log_file_path)} - Epoch {current_epoch}: valid acc = {valid_acc:.6f}")
    
    return epochs, valid_accs

def segment_data(epochs, valid_accs, segments):
    """
    将数据按epoch段分割
    segments: list of tuples [(start, end), ...]
    返回: list of [(epochs, valid_accs), ...]
    """
    segmented_data = []
    
    for start, end in segments:
        seg_epochs = []
        seg_accs = []
        
        for i, epoch in enumerate(epochs):
            if start <= epoch <= end:
                seg_epochs.append(epoch)
                seg_accs.append(valid_accs[i])
        
        segmented_data.append((seg_epochs, seg_accs))
    
    return segmented_data

def compare_and_plot_segmented(log_file_paths, labels=None, segments=None):
    """
    从多个日志文件中提取真实valid acc并按epoch段绘制对比曲线
    segments: list of tuples, e.g., [(0, 50), (51, 100), (101, 150)]
    """
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(log_file_paths))]
    
    if segments is None:
        segments = [(0, 50), (51, 100), (101, 150)]
    
    all_data = []
    
    # 提取所有文件的数据
    for i, log_file_path in enumerate(log_file_paths):
        print(f"\n正在处理文件 {i+1}: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            print(f"错误: 日志文件不存在: {log_file_path}")
            continue
            
        epochs, valid_accs = extract_valid_acc_from_log(log_file_path)
        
        if epochs and valid_accs:
            # 分段处理数据
            segmented = segment_data(epochs, valid_accs, segments)
            all_data.append((epochs, valid_accs, segmented, labels[i]))
            print(f"成功提取 {len(epochs)} 个数据点")
        else:
            print(f"未找到任何valid acc数据")
    
    if not all_data:
        print("没有成功提取到任何数据")
        return
    
    # 定义颜色和样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # 创建图形 - 3行2列
    fig = plt.figure(figsize=(18, 14))
    
    # 为每个segment创建对比图
    for seg_idx, (start, end) in enumerate(segments):
        # 主对比图
        ax1 = plt.subplot(3, 2, seg_idx * 2 + 1)
        
        for exp_idx, (epochs, best_accs, segmented, label) in enumerate(all_data):
            seg_epochs, seg_accs = segmented[seg_idx]
            
            if seg_epochs and seg_accs:
                color = colors[exp_idx % len(colors)]
                marker = markers[exp_idx % len(markers)]
                ax1.plot(seg_epochs, seg_accs, color=color, linewidth=2, marker=marker, 
                        markersize=4, label=label, alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Valid Accuracy (%)', fontsize=11)
        
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # 设置y轴范围
        all_seg_accs = []
        for _, _, segmented, _ in all_data:
            all_seg_accs.extend(segmented[seg_idx][1])
        
        if all_seg_accs:
            min_acc = min(all_seg_accs)
            max_acc = max(all_seg_accs)
            margin = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 1
            ax1.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))
        
        # 详细对比图（带数值标注）
        ax2 = plt.subplot(3, 2, seg_idx * 2 + 2)
        
        for exp_idx, (epochs, best_accs, segmented, label) in enumerate(all_data):
            seg_epochs, seg_accs = segmented[seg_idx]
            
            if seg_epochs and seg_accs:
                color = colors[exp_idx % len(colors)]
                marker = markers[exp_idx % len(markers)]
                ax2.plot(seg_epochs, seg_accs, color=color, linewidth=2, marker=marker, 
                        markersize=5, label=label, alpha=0.8)
                
                # 添加关键点标注
                step = max(1, len(seg_epochs) // 5)
                for j in range(0, len(seg_epochs), step):
                    if j < len(seg_epochs):
                        ax2.annotate(f'{seg_accs[j]:.2f}', 
                                    (seg_epochs[j], seg_accs[j]), 
                                    textcoords="offset points", 
                                    xytext=(0, 8), 
                                    ha='center',
                                    fontsize=7,
                                    color=color,
                                    alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Valid Accuracy (%)', fontsize=11)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        
        # 设置y轴范围
        if all_seg_accs:
            ax2.set_ylim(max(0, min_acc - margin), min(100, max_acc + margin))
    
    plt.tight_layout()
    
    # 保存图片
    base_dir = os.path.dirname(log_file_paths[0])
    plot_save_path = os.path.join(base_dir, "valid_acc_comparison_segmented.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\n分段对比图已保存到: {plot_save_path}")
    
    plt.show()
    
    # 打印详细统计信息
    print(f"\n{'='*80}")
    print(f"详细统计信息（分段）")
    print(f"{'='*80}")
    
    for exp_idx, (epochs, valid_accs, segmented, label) in enumerate(all_data):
        print(f"\n{label}:")
        print(f"  总epoch数: {len(epochs)}")
        print(f"  整体最高准确率: {max(valid_accs):.6f}%")
        print(f"  整体最终准确率: {valid_accs[-1]:.6f}%")
        
        for seg_idx, (start, end) in enumerate(segments):
            seg_epochs, seg_accs = segmented[seg_idx]
            if seg_accs:
                print(f"\n  Epoch {start}-{end}:")
                print(f"    数据点数: {len(seg_epochs)}")
                print(f"    最高准确率: {max(seg_accs):.6f}%")
                print(f"    最低准确率: {min(seg_accs):.6f}%")
                print(f"    平均准确率: {np.mean(seg_accs):.6f}%")
                print(f"    该段最终准确率: {seg_accs[-1]:.6f}%")
                
                # 准确率提升
                if len(seg_accs) > 1:
                    improvement = seg_accs[-1] - seg_accs[0]
                    print(f"    准确率提升: {improvement:+.6f}%")
    
    # 各段对比
    print(f"\n{'='*80}")
    print(f"各段对比")
    print(f"{'='*80}")
    
    for seg_idx, (start, end) in enumerate(segments):
        print(f"\nEpoch {start}-{end}:")
        
        seg_max_accs = []
        seg_final_accs = []
        
        for exp_idx, (epochs, valid_accs, segmented, label) in enumerate(all_data):
            seg_epochs, seg_accs = segmented[seg_idx]
            if seg_accs:
                seg_max_accs.append((max(seg_accs), label))
                seg_final_accs.append((seg_accs[-1], label))
        
        if seg_max_accs:
            best_max = max(seg_max_accs, key=lambda x: x[0])
            print(f"  该段最高准确率: {best_max[1]} ({best_max[0]:.6f}%)")
        
        if seg_final_accs:
            best_final = max(seg_final_accs, key=lambda x: x[0])
            print(f"  该段最终准确率: {best_final[1]} ({best_final[0]:.6f}%)")
    
    return all_data

# 主程序
if __name__ == "__main__":
    # 您的两个日志文件路径
    log_file_paths = [
        r"D:\dvsc10_new0\cifar10\k0.1-8layers-4step-150epoch\log.txt",
        r"D:\dvsc10_new0\cifar10\k0-8layers-4step-150epoch\log.txt"
    ]
    
    # 自定义标签（可选）
    labels = [
        "DIF神经元 k=0.1",
        "LIF神经元"
    ]
    
    # 定义epoch段
    segments = [(0, 50), (51, 100), (101, 150)]
    
    print("正在对比训练日志的真实valid acc曲线（分段展示）...")
    print(f"分段设置: {segments}")
    
    # 检查文件是否存在
    missing_files = []
    for i, log_file_path in enumerate(log_file_paths):
        if not os.path.exists(log_file_path):
            missing_files.append(f"文件 {i+1}: {log_file_path}")
    
    if missing_files:
        print("错误: 以下文件不存在:")
        for missing_file in missing_files:
            print(f"  {missing_file}")
        print("请检查文件路径是否正确")
    else:
        # 提取数据并绘制对比曲线
        all_data = compare_and_plot_segmented(log_file_paths, labels, segments)

