#!/usr/bin/env python3
"""
简化版：从训练日志中提取best valid acc并绘制曲线
"""

import re
import matplotlib.pyplot as plt
import numpy as np
import os

def extract_and_plot_best_acc(log_file_path):
    """
    从日志文件中提取best valid acc并绘制曲线
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
    
    # 提取所有epoch和对应的best valid acc
    epochs = []
    best_accs = []
    
    lines = content.split('\n')
    current_epoch = 0
    
    for line in lines:
        # 查找epoch信息
        epoch_match = re.search(r'epoch\s+(\d+)\s*/\s*(\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # 查找best valid acc信息
        acc_match = re.search(r'best valid acc ([\d.]+)', line)
        if acc_match:
            best_acc = float(acc_match.group(1))
            epochs.append(current_epoch)
            best_accs.append(best_acc)
            print(f"Epoch {current_epoch}: best valid acc = {best_acc:.6f}")
    
    if not epochs:
        print("未找到任何best valid acc数据")
        return
    
    # 绘制曲线
    plt.figure(figsize=(12, 8))
    
    # 主图：best valid acc曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs, best_accs, 'b-', linewidth=2, marker='o', markersize=4, label='Best Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Best Valid Accuracy (%)')
    plt.title('Best Valid Accuracy Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 设置y轴范围
    if best_accs:
        min_acc = min(best_accs)
        max_acc = max(best_accs)
        margin = (max_acc - min_acc) * 0.1
        plt.ylim(max(0, min_acc - margin), min(100, max_acc + margin))
    
    # 子图：详细曲线（带数值标注）
    plt.subplot(2, 1, 2)
    plt.plot(epochs, best_accs, 'r-', linewidth=2, marker='s', markersize=5, label='Best Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Best Valid Accuracy (%)')
    plt.title('Best Valid Accuracy Detailed Curve (with values)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加数值标注
    for i in range(0, len(epochs), max(1, len(epochs)//10)):  # 最多标注10个点
        if i < len(epochs) and i < len(best_accs):
            plt.annotate(f'{best_accs[i]:.2f}%', 
                        (epochs[i], best_accs[i]), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.tight_layout()
    
    # 保存图片
    base_dir = os.path.dirname(log_file_path)
    plot_save_path = os.path.join(base_dir, "best_acc_curve.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存到: {plot_save_path}")
    
    plt.show()
    
    # 打印统计信息
    print(f"\n=== 统计信息 ===")
    print(f"总epoch数: {len(epochs)}")
    print(f"最高准确率: {max(best_accs):.6f}%")
    print(f"最低准确率: {min(best_accs):.6f}%")
    print(f"平均准确率: {np.mean(best_accs):.6f}%")
    print(f"最终准确率: {best_accs[-1]:.6f}%")
    
    # 找到最高准确率对应的epoch
    max_idx = np.argmax(best_accs)
    print(f"最高准确率出现在Epoch {epochs[max_idx]}: {best_accs[max_idx]:.6f}%")
    
    return epochs, best_accs

# 主程序
if __name__ == "__main__":
    # 您的日志文件路径
    log_file_path = r"F:\data\floyed\darts\logs\eval\dvsc10_new0\cifar10\eval-EXP-20251022-2236-\log.txt"
    
    print(f"正在读取日志文件: {log_file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件不存在: {log_file_path}")
        print("请检查文件路径是否正确")
    else:
        # 提取数据并绘制曲线
        epochs, best_accs = extract_and_plot_best_acc(log_file_path)
