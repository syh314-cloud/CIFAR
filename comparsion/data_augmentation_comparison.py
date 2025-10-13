"""
数据增强对比实验
第一组：单因素实验 - 测试每种数据增强方法的效果（包含Baseline）
第二组：贪心叠加实验 - 基于第一组最优结果进行贪心式叠加
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as np
    import numpy as np_cpu
except ImportError:
    import numpy as np
    np_cpu = np
import matplotlib
matplotlib.use('Agg')  # 设置后端，避免GUI问题
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# 设置中文字体和显示参数
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 修改工作目录到项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

from utils.data_loader import train_images, train_labels, val_images, val_labels, test_images, test_labels
from utils.data_augmentation import random_flip, random_crop, random_rotate, random_noise
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import Adam

class DataAugmentationExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        self.epochs = 100
        
        # 贪心叠加的阈值设置
        self.delta_threshold = 0.003  # 验证集准确率增益阈值（0.3%）
        
        # 定义所有可用的数据增强方法
        self.augmentation_methods = {
            'Flip': lambda x: random_flip(x, prob=0.5),
            'Crop': lambda x: random_crop(x, crop_size=32, padding=4),
            'Rotate': lambda x: random_rotate(x, max_angle=15),
            'Noise': lambda x: random_noise(x, std=0.02)
        }
        
        self.results_single = {}  # 第一组：单因素实验结果
        self.results_greedy = []  # 第二组：贪心叠加路径
        self.greedy_details = []  # 贪心叠加的详细信息
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组用于matplotlib"""
        if hasattr(data, 'get'):  # CuPy数组
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data

    def _apply_augmentations(self, x, aug_list):
        """应用一系列数据增强"""
        if not aug_list:
            return x
        
        # 重塑为4D数组
        x = x.reshape(-1, 3, 32, 32)
        
        # 按顺序应用所有增强
        for aug_name in aug_list:
            if aug_name in self.augmentation_methods:
                x = self.augmentation_methods[aug_name](x.copy())
        
        # 重新扁平化
        return x.reshape(x.shape[0], -1)

    def train_single_model(self, aug_name, aug_function, description=""):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - {description}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        step_losses = []
        val_accuracies = []
        
        # 关闭所有Dropout
        model.dropout1.p = 0.0
        model.dropout2.p = 0.0
        model.dropout3.p = 0.0
        
        for epoch in range(self.epochs):
            np.random.seed(self.SEED + epoch)
            idx = np.random.permutation(train_images.shape[0])
            shuffled_images = train_images[idx]
            shuffled_labels = train_labels[idx]
            
            epoch_losses = []
            
            for i in range(0, shuffled_images.shape[0], self.batch_size):
                x = shuffled_images[i:i+self.batch_size]
                y = shuffled_labels[i:i+self.batch_size]
                
                # 应用数据增强
                if aug_function is not None:
                    x = aug_function(x)
                
                model.zero_grad()
                y_pred = model.forward(x, training=False)
          
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=0.0)
                step_losses.append(loss)
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                optimizer.step(model)
            
            # 验证准确率
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(float(val_acc))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = float(np.mean(np.array(epoch_losses)))
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 测试准确率
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        # 训练集准确率
        train_pred = model.forward(train_images, training=False)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(train_labels, axis=1))
        
        print(f"✅ 训练完成 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        return {
            'augmentation_name': aug_name,
            'description': description,
            'step_losses': [float(x) for x in self._to_numpy(step_losses)],
            'val_accuracies': [float(x) for x in val_accuracies],
            'test_accuracy': float(test_acc),
            'train_accuracy': float(train_acc),
            'best_val_accuracy': float(max(val_accuracies)),
            'steps_per_epoch': len(train_images) // self.batch_size
        }
    
    def run_single_factor_experiments(self):
        """第一组：单因素实验"""
        print("="*80)
        print("第一组实验：单因素实验（Phase A）")
        print("="*80)
        
        # 1. Baseline (无数据增强)
        print("\n测试 Baseline...")
        result = self.train_single_model(
            'Baseline',
            None,
            "Baseline (无数据增强)"
        )
        self.results_single['Baseline'] = result
        
        # 2. 测试每种单独的数据增强方法
        for aug_name, aug_func in self.augmentation_methods.items():
            print(f"\n测试 {aug_name}...")
            
            def make_aug_wrapper(func):
                def wrapper(x):
                    x = x.reshape(-1, 3, 32, 32)
                    x = func(x.copy())
                    return x.reshape(x.shape[0], -1)
                return wrapper
            
            result = self.train_single_model(
                aug_name,
                make_aug_wrapper(aug_func),
                aug_name
            )
            self.results_single[aug_name] = result
        
        # 排序并显示结果
        sorted_results = sorted(
            self.results_single.items(),
            key=lambda x: x[1]['best_val_accuracy'],
            reverse=True
        )
        
        print(f"\n{'='*80}")
        print("Phase A 结果排序（按验证集准确率）：")
        print(f"{'='*80}")
        for rank, (name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {name:10s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        print(f"{'='*80}\n")
    
    def run_greedy_combination_experiments(self):
        """第二组：贪心叠加实验"""
        print("="*80)
        print("第二组实验：贪心叠加实验（Phase B）")
        print("="*80)
        
        # 获取除Baseline外的增强方法，按验证集准确率排序
        augmentation_results = {
            name: result for name, result in self.results_single.items()
            if name != 'Baseline'
        }
        
        sorted_augs = sorted(
            augmentation_results.items(),
            key=lambda x: x[1]['best_val_accuracy'],
            reverse=True
        )
        
        print("\n增强方法排序（按Phase A验证集准确率）：")
        for rank, (name, result) in enumerate(sorted_augs, 1):
            print(f"  {rank}. {name}: {result['best_val_accuracy']:.4f}")
        
        # 初始化：从Baseline开始
        current_combination = []
        baseline_val_acc = self.results_single['Baseline']['best_val_accuracy']
        current_val_acc = baseline_val_acc
        
        # 记录G1: Baseline
        self.greedy_details.append({
            'step': 'G1',
            'combination': [],
            'combination_name': 'Baseline',
            'val_accuracy': baseline_val_acc,
            'test_accuracy': self.results_single['Baseline']['test_accuracy'],
            'delta_acc': 0.0,
            'description': 'Baseline (无数据增强)'
        })
        
        # 可选的增强方法池
        remaining_augs = [name for name, _ in sorted_augs]
        step_num = 2
        
        print(f"\n开始贪心叠加过程...")
        print(f"G1: Baseline - Val Acc={baseline_val_acc:.4f}")
        
        while remaining_augs:
            print(f"\n{'='*60}")
            print(f"尝试第 {step_num} 步叠加（当前组合: {'+'.join(current_combination) if current_combination else 'Baseline'}）")
            print(f"{'='*60}")
            
            best_next_aug = None
            best_next_val_acc = current_val_acc
            best_next_result = None
            
            # 尝试每个剩余的增强方法
            for candidate_aug in remaining_augs:
                test_combination = current_combination + [candidate_aug]
                test_name = ' + '.join(test_combination)
                
                print(f"\n尝试: {test_name}")
                
                # 训练这个组合
                def make_combined_aug(aug_list):
                    def wrapper(x):
                        return self._apply_augmentations(x, aug_list)
                    return wrapper
                
                result = self.train_single_model(
                    test_name,
                    make_combined_aug(test_combination),
                    test_name
                )
                
                # 检查是否有改进
                delta_acc = result['best_val_accuracy'] - current_val_acc
                print(f"  结果: Val Acc={result['best_val_accuracy']:.4f}, ΔAcc={delta_acc:+.4f}")
                
                if result['best_val_accuracy'] > best_next_val_acc:
                    best_next_val_acc = result['best_val_accuracy']
                    best_next_aug = candidate_aug
                    best_next_result = result
            
            # 检查最佳候选是否满足增益阈值
            if best_next_aug is not None:
                delta_acc = best_next_val_acc - current_val_acc
                
                if delta_acc >= self.delta_threshold:
                    # 接受这个增强
                    current_combination.append(best_next_aug)
                    current_val_acc = best_next_val_acc
                    remaining_augs.remove(best_next_aug)
                    
                    step_name = f"G{step_num}"
                    combination_name = ' + '.join(current_combination)
                    
                    self.greedy_details.append({
                        'step': step_name,
                        'combination': current_combination.copy(),
                        'combination_name': combination_name,
                        'val_accuracy': best_next_val_acc,
                        'test_accuracy': best_next_result['test_accuracy'],
                        'delta_acc': delta_acc,
                        'description': f"添加 {best_next_aug}",
                        'full_result': best_next_result
                    })
                    
                    print(f"\n✅ {step_name}: {combination_name}")
                    print(f"   Val Acc={best_next_val_acc:.4f}, ΔAcc={delta_acc:+.4f} >= {self.delta_threshold:.4f} (接受)")
                    
                    step_num += 1
                else:
                    # 增益不足，停止叠加
                    print(f"\n⛔ 最佳增益 ΔAcc={delta_acc:+.4f} < {self.delta_threshold:.4f}，停止叠加")
                    break
            else:
                # 没有找到改进，停止
                print(f"\n⛔ 无法找到改进方案，停止叠加")
                break
        
        print(f"\n{'='*80}")
        print("贪心叠加完成！")
        print(f"{'='*80}\n")
    
    def run_experiments(self):
        """运行所有实验"""
        print("="*80)
        print("数据增强对比实验")
        print("="*80)
        print(f"固定参数:")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - L2 Regularization: 0.0 (关闭)")
        print(f"  - Dropout: 0.0 (关闭)")
        print(f"  - Delta Threshold: {self.delta_threshold:.4f} ({self.delta_threshold*100:.2f}%)")
        print("="*80)
        
        # Phase A: 单因素实验
        self.run_single_factor_experiments()
        
        # Phase B: 贪心叠加实验
        self.run_greedy_combination_experiments()
        
        self.save_results()
        self.plot_results()
        self._print_summary()
    
    def save_results(self):
        """保存实验结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_augmentation_comparison_results_{timestamp}.json"
        
        save_data = {
            'experiment_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'seed': self.SEED,
                'l2_regularization': 0.0,
                'dropout': 0.0,
                'delta_threshold': self.delta_threshold
            },
            'results_single': self.results_single,
            'greedy_path': self.greedy_details
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ========== 第一组图表：单因素实验结果 ==========
        fig1 = plt.figure(figsize=(18, 5))
        
        colors_single = ['#95a5a6', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # Baseline是灰色
        
        # 1.1 验证准确率曲线
        ax1 = plt.subplot(1, 3, 1)
        sorted_names = sorted(self.results_single.keys(), 
                            key=lambda x: self.results_single[x]['best_val_accuracy'],
                            reverse=True)
        
        for idx, aug_name in enumerate(sorted_names):
            result = self.results_single[aug_name]
            ax1.plot(range(1, self.epochs + 1), 
                    result['val_accuracies'],
                    label=aug_name,
                    color=colors_single[idx % len(colors_single)],
                    linewidth=2, alpha=0.8,
                    linestyle='--' if aug_name == 'Baseline' else '-')
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Validation Accuracy', fontsize=11)
        ax1.set_title('Phase A: Single Factor Validation Accuracy', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 1.2 验证集和测试集准确率柱状图
        ax2 = plt.subplot(1, 3, 2)
        
        aug_names = sorted_names
        val_accs = [self.results_single[name]['best_val_accuracy'] for name in aug_names]
        test_accs = [self.results_single[name]['test_accuracy'] for name in aug_names]
        
        x = self._to_numpy(np.arange(len(aug_names)))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, val_accs, width, 
                       label='Validation Accuracy',
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax2.bar(x + width/2, test_accs, width,
                       label='Test Accuracy',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=7)
        
        ax2.set_xlabel('Augmentation Method', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Phase A: Val vs Test Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(aug_names, rotation=20, ha='right', fontsize=9)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_ylim([0, 1.0])
        
        # 1.3 准确率提升对比（相对于Baseline）
        ax3 = plt.subplot(1, 3, 3)
        
        baseline_test_acc = self.results_single['Baseline']['test_accuracy']
        improvements = []
        improvement_names = []
        
        for name in aug_names:
            if name != 'Baseline':
                improvement = self.results_single[name]['test_accuracy'] - baseline_test_acc
                improvements.append(improvement)
                improvement_names.append(name)
        
        x_imp = self._to_numpy(np.arange(len(improvement_names)))
        colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
        
        bars = ax3.bar(x_imp, improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.4f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Augmentation Method', fontsize=11)
        ax3.set_ylabel('Test Accuracy Improvement vs Baseline', fontsize=11)
        ax3.set_title('Phase A: Improvement over Baseline', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_imp)
        ax3.set_xticklabels(improvement_names, rotation=20, ha='right', fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        
        plot_filename1 = f"data_augmentation_single_plots_{timestamp}.png"
        plt.savefig(plot_filename1, bbox_inches='tight')
        print(f"📊 Phase A 图表已保存到: {plot_filename1}")
        plt.close()
        
        # ========== 第二组图表：贪心叠加路径 ==========
        if len(self.greedy_details) > 1:
            fig2 = plt.figure(figsize=(18, 5))
            
            # 2.1 贪心路径的准确率变化
            ax4 = plt.subplot(1, 3, 1)
            
            steps = [detail['step'] for detail in self.greedy_details]
            val_accs = [detail['val_accuracy'] for detail in self.greedy_details]
            test_accs = [detail['test_accuracy'] for detail in self.greedy_details]
            
            x_steps = range(len(steps))
            
            ax4.plot(x_steps, val_accs, marker='o', label='Validation Accuracy', 
                    color='#9b59b6', linewidth=2, markersize=8)
            ax4.plot(x_steps, test_accs, marker='s', label='Test Accuracy',
                    color='#e74c3c', linewidth=2, markersize=8)
            
            # 添加数值标签
            for i, (val, test) in enumerate(zip(val_accs, test_accs)):
                ax4.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=8)
                ax4.text(i, test, f'{test:.4f}', ha='center', va='top', fontsize=8)
            
            ax4.set_xlabel('Greedy Step', fontsize=11)
            ax4.set_ylabel('Accuracy', fontsize=11)
            ax4.set_title('Phase B: Greedy Combination Path', fontsize=12, fontweight='bold')
            ax4.set_xticks(x_steps)
            ax4.set_xticklabels(steps, fontsize=10)
            ax4.legend(loc='best', framealpha=0.9)
            ax4.grid(True, alpha=0.3, linestyle='--')
            
            # 2.2 边际收益 ΔAcc
            ax5 = plt.subplot(1, 3, 2)
            
            delta_accs = [detail['delta_acc'] for detail in self.greedy_details[1:]]  # 跳过G1
            delta_steps = steps[1:]
            
            x_delta = range(len(delta_steps))
            colors_delta = ['#2ecc71' if d >= self.delta_threshold else '#f39c12' for d in delta_accs]
            
            bars = ax5.bar(x_delta, delta_accs, color=colors_delta, alpha=0.8, edgecolor='black', linewidth=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:+.4f}',
                        ha='center', va='bottom', fontsize=9)
            
            ax5.axhline(y=self.delta_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={self.delta_threshold:.4f}')
            ax5.set_xlabel('Greedy Step', fontsize=11)
            ax5.set_ylabel('ΔAcc (Marginal Gain)', fontsize=11)
            ax5.set_title('Phase B: Marginal Accuracy Gain', fontsize=12, fontweight='bold')
            ax5.set_xticks(x_delta)
            ax5.set_xticklabels(delta_steps, fontsize=10)
            ax5.legend(loc='best', framealpha=0.9)
            ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # 2.3 最终对比：Baseline vs 最优组合
            ax6 = plt.subplot(1, 3, 3)
            
            comparison_names = ['Baseline', 'Greedy Best']
            comparison_val = [
                self.greedy_details[0]['val_accuracy'],
                self.greedy_details[-1]['val_accuracy']
            ]
            comparison_test = [
                self.greedy_details[0]['test_accuracy'],
                self.greedy_details[-1]['test_accuracy']
            ]
            
            x_comp = self._to_numpy(np.arange(len(comparison_names)))
            width = 0.35
            
            bars1 = ax6.bar(x_comp - width/2, comparison_val, width,
                           label='Validation Accuracy',
                           color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
            bars2 = ax6.bar(x_comp + width/2, comparison_test, width,
                           label='Test Accuracy',
                           color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=9)
            
            ax6.set_ylabel('Accuracy', fontsize=11)
            ax6.set_title('Phase B: Baseline vs Greedy Best', fontsize=12, fontweight='bold')
            ax6.set_xticks(x_comp)
            ax6.set_xticklabels(comparison_names, fontsize=10)
            ax6.legend(loc='best', framealpha=0.9)
            ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax6.set_ylim([0, 1.0])
            
            plt.tight_layout()
            
            plot_filename2 = f"data_augmentation_greedy_plots_{timestamp}.png"
            plt.savefig(plot_filename2, bbox_inches='tight')
            print(f"📊 Phase B 图表已保存到: {plot_filename2}")
            plt.close()
    
    def _print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*80}")
        print("实验总结")
        print(f"{'='*80}")
        
        print("\n📋 Phase A: 单因素实验结果")
        print("-"*80)
        sorted_single = sorted(
            self.results_single.items(),
            key=lambda x: x[1]['best_val_accuracy'],
            reverse=True
        )
        for rank, (name, result) in enumerate(sorted_single, 1):
            print(f"  {rank}. {name:10s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        print(f"\n📋 Phase B: 贪心叠加路径")
        print("-"*80)
        for detail in self.greedy_details:
            step = detail['step']
            name = detail['combination_name']
            val_acc = detail['val_accuracy']
            test_acc = detail['test_accuracy']
            delta = detail['delta_acc']
            
            if step == 'G1':
                print(f"  {step}: {name:30s} - Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")
            else:
                print(f"  {step}: {name:30s} - Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}, ΔAcc={delta:+.4f}")
        
        # 总结贪心叠加的效果
        if len(self.greedy_details) > 1:
            baseline_test = self.greedy_details[0]['test_accuracy']
            final_test = self.greedy_details[-1]['test_accuracy']
            total_gain = final_test - baseline_test
            
            print(f"\n🎯 贪心叠加总结:")
            print(f"   起点 (Baseline):          Test Acc = {baseline_test:.4f}")
            print(f"   终点 (最优组合):          Test Acc = {final_test:.4f}")
            print(f"   总增益:                   ΔAcc = {total_gain:+.4f} ({total_gain*100:+.2f}%)")
            print(f"   叠加步数:                 {len(self.greedy_details) - 1} 步")
            print(f"   最优组合:                 {self.greedy_details[-1]['combination_name']}")
            
            if total_gain > 0:
                print(f"   ✅ 叠加增强带来了显著提升！")
            else:
                print(f"   ⚠️ 叠加增强未带来提升")
        
        print(f"{'='*80}\n")

def main():
    experiment = DataAugmentationExperiment()
    experiment.run_experiments()

if __name__ == "__main__":
    main()
