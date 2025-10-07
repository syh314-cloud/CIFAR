#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本 - 测试已训练模型在测试集上的性能
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import test_images, test_labels
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.optimizers import Momentum
from utils.metrics import (
    accuracy, top_k_accuracy, confusion_matrix, 
    classification_report, print_confusion_matrix, CIFAR10_CLASSES
)

def evaluate_test_set():
    """
    评估测试集性能
    注意：需要先运行train.py训练模型，这里会重新创建相同的模型结构
    """
    print("🎯 测试集评估")
    print("="*50)
    
    # 创建与train.py相同的模型结构
    model = MLP(test_images.shape[1], 1024, 512, 256, 10)
    
    # 注意：这里应该加载训练好的权重，但由于没有保存功能，
    # 实际使用时需要先运行train.py或者添加模型保存/加载功能
    print("⚠️  注意：当前使用随机初始化的权重")
    print("   实际使用时应该加载train.py训练好的模型权重")
    print()
    
    # 设置为评估模式（关闭dropout）
    model.dropout1.p = 0.0
    model.dropout2.p = 0.0
    model.dropout3.p = 0.0
    
    # 获取测试集预测
    print("🔮 生成测试集预测...")
    test_pred = model.forward(test_images, training=False)
    
    # 计算各项指标
    print("\n📊 测试集性能指标:")
    print("-" * 40)
    
    # 1. 准确率
    test_acc = accuracy(test_pred, test_labels)
    print(f"准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # 2. Top-5准确率
    top5_acc = top_k_accuracy(test_pred, test_labels, k=5)
    print(f"Top-5准确率: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    
    # 3. 损失
    loss_fn = CrossEntropyLoss()
    test_loss = loss_fn.forward(test_pred, test_labels)
    print(f"交叉熵损失: {test_loss:.4f}")
    
    # 4. 混淆矩阵
    print(f"\n📊 混淆矩阵:")
    cm = confusion_matrix(test_pred, test_labels)
    print_confusion_matrix(cm, CIFAR10_CLASSES)
    
    # 5. 详细分类报告
    print(f"\n📋 详细分类报告:")
    report = classification_report(test_pred, test_labels, CIFAR10_CLASSES)
    
    # 6. 各类别准确率
    print(f"\n📈 各类别准确率:")
    print("-" * 30)
    pred_labels = np.argmax(test_pred, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    for i, class_name in enumerate(CIFAR10_CLASSES):
        class_mask = (true_labels == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(pred_labels[class_mask] == true_labels[class_mask])
            print(f"{class_name:<12}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # 7. 总结
    print(f"\n🎯 测试集评估总结:")
    print("="*30)
    print(f"测试样本数: {len(test_images)}")
    print(f"总体准确率: {test_acc*100:.2f}%")
    print(f"Top-5准确率: {top5_acc*100:.2f}%")
    
    # 性能评级
    if test_acc >= 0.6:
        grade = "优秀 🏆"
    elif test_acc >= 0.5:
        grade = "良好 👍"
    elif test_acc >= 0.4:
        grade = "一般 📈"
    else:
        grade = "需要改进 ⚠️"
    
    print(f"性能评级: {grade}")
    
    return {
        'accuracy': test_acc,
        'top5_accuracy': top5_acc,
        'loss': test_loss,
        'confusion_matrix': cm,
        'predictions': test_pred
    }

if __name__ == "__main__":
    print("🧪 神经网络测试集评估")
    print(f"测试集形状: {test_images.shape}")
    print(f"标签形状: {test_labels.shape}")
    print(f"类别数: {len(CIFAR10_CLASSES)}")
    print()
    
    results = evaluate_test_set()
    
    print(f"\n✅ 评估完成！")
    print(f"测试准确率: {results['accuracy']*100:.2f}%")