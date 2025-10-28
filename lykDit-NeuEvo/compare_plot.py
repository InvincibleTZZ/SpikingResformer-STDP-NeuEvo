#!/usr/bin/env python3
"""
对比版：从两个训练日志中提取当前epoch的真实valid acc并绘制对比曲线
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

def compare_and_plot_valid_acc(log_file_paths, labels=None):
    """
    从多个日志文件中提取真实valid acc并绘制对比曲线
    """
    if labels is None:
        labels = [f"Experiment {i+1}" for i in range(len(log_file_paths))]
    
    all_data = []
    
    # 提取所有文件的数据
    for i, log_file_path in enumerate(log_file_paths):
        print(f"\n正在处理文件 {i+1}: {log_file_path}")
        
        if not os.path.exists(log_file_path):
            print(f"错误: 日志文件不存在: {log_file_path}")
            continue
            
        epochs, valid_accs = extract_valid_acc_from_log(log_file_path)
        
        if epochs and valid_accs:
            all_data.append((epochs, valid_accs, labels[i]))
            print(f"成功提取 {len(epochs)} 个数据点")
        else:
            print(f"未找到任何valid acc数据")
    
    if not all_data:
        print("没有成功提取到任何数据")
        return
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 定义颜色和样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    
    # 主图：对比曲线
    plt.subplot(2, 1, 1)
    for i, (epochs, valid_accs, label) in enumerate(all_data):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(epochs, valid_accs, color=color, linewidth=2, marker=marker, 
                markersize=4, label=label, alpha=0.8)
    
    plt.xlabel('Epoch')
    plt.ylabel('Valid Accuracy (%)')
    plt.title('Valid Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置y轴范围
    all_accs = [acc for _, accs, _ in all_data for acc in accs]
    if all_accs:
        min_acc = min(all_accs)
        max_acc = max(all_accs)
        margin = (max_acc - min_acc) * 0.1
        plt.ylim(max(0, min_acc - margin), min(100, max_acc + margin))
    
    # 子图1：详细对比（带数值标注）
    plt.subplot(2, 1, 2)
    for i, (epochs, valid_accs, label) in enumerate(all_data):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(epochs, valid_accs, color=color, linewidth=2, marker=marker, 
                markersize=5, label=label, alpha=0.8)
        
        # 添加关键点标注（每5个点标注一次）
        for j in range(0, len(epochs), max(1, len(epochs)//5)):
            if j < len(epochs) and j < len(valid_accs):
                plt.annotate(f'{valid_accs[j]:.2f}%', 
                            (epochs[j], valid_accs[j]), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=7,
                            color=color)
    
    plt.xlabel('Epoch')
    plt.ylabel('Valid Accuracy (%)')
    plt.title('Detailed Comparison (with values)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 子图2：最终准确率对比柱状图
    
    
    # 子图3：最高准确率对比
   
    
    
    
    
    
    plt.tight_layout()
    
    # 保存图片
    base_dir = os.path.dirname(log_file_paths[0])
    plot_save_path = os.path.join(base_dir, "valid_acc_comparison.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\n对比图已保存到: {plot_save_path}")
    
    plt.show()
    
    # 打印详细统计信息
    print(f"\n=== 详细统计信息 ===")
    for i, (epochs, valid_accs, label) in enumerate(all_data):
        print(f"\n{label}:")
        print(f"  总epoch数: {len(epochs)}")
        print(f"  最高准确率: {max(valid_accs):.6f}%")
        print(f"  最低准确率: {min(valid_accs):.6f}%")
        print(f"  平均准确率: {np.mean(valid_accs):.6f}%")
        print(f"  最终准确率: {valid_accs[-1]:.6f}%")
        
        # 找到最高准确率对应的epoch
        max_idx = np.argmax(valid_accs)
        print(f"  最高准确率出现在Epoch {epochs[max_idx]}: {valid_accs[max_idx]:.6f}%")
    
    # 整体对比
    print(f"\n=== 整体对比 ===")
    final_accs = [valid_accs[-1] for _, valid_accs, _ in all_data]
    max_accs = [max(valid_accs) for _, valid_accs, _ in all_data]
    
    best_final_idx = np.argmax(final_accs)
    best_max_idx = np.argmax(max_accs)
    
    print(f"最终准确率最高: {all_data[best_final_idx][2]} ({final_accs[best_final_idx]:.6f}%)")
    print(f"最高准确率最高: {all_data[best_max_idx][2]} ({max_accs[best_max_idx]:.6f}%)")
    
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
        "DIF神经元 k=0.1--8layers--2step--32init_channels--50epochs  ",
        "LIF神经元 8layers--2step--32init_channels--50epochs"
    ]
    
    print("正在对比两个训练日志的真实valid acc曲线...")
    
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
        all_data = compare_and_plot_valid_acc(log_file_paths, labels)
