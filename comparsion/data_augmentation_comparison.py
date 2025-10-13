"""
数据增强对比实验
批量训练模型，对比不同数据增强方法的效果
测试方法：Flip、Crop、Rotate、Noise、无数据增强（Baseline）
生成训练Loss曲线、验证Accuracy曲线和测试准确率柱状图
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
        
        # 要测试的数据增强方法 - 分为两组实验
        # 第一组：单独测试每种方法
        self.augmentation_configs_single = {
            'Flip': {
                'name': 'Flip',
                'function': lambda x: random_flip(x, prob=0.5)
            },
            'Crop': {
                'name': 'Crop',
                'function': lambda x: random_crop(x, crop_size=32, padding=4)
            },
            'Rotate': {
                'name': 'Rotate',
                'function': lambda x: random_rotate(x, max_angle=15)
            },
            'Noise': {
                'name': 'Noise',
                'function': lambda x: random_noise(x, std=0.02)
            }
        }
        
        # 第二组：叠加测试
        self.augmentation_configs_combined = {
            'Flip_Only': {
                'name': 'Flip',
                'function': lambda x: random_flip(x, prob=0.5)
            },
            'Flip_Crop': {
                'name': 'Flip + Crop',
                'function': lambda x: random_crop(random_flip(x, prob=0.5), crop_size=32, padding=4)
            },
            'Flip_Crop_Rotate': {
                'name': 'Flip + Crop + Rotate',
                'function': lambda x: random_rotate(random_crop(random_flip(x, prob=0.5), crop_size=32, padding=4), max_angle=15)
            },
            'Flip_Crop_Rotate_Noise': {
                'name': 'Flip + Crop + Rotate + Noise',
                'function': lambda x: random_noise(random_rotate(random_crop(random_flip(x, prob=0.5), crop_size=32, padding=4), max_angle=15), std=0.02)
            }
        }
        
        self.results_single = {}  # 第一组：单独测试结果
        self.results_combined = {}  # 第二组：叠加测试结果
        self.epochs = 100
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组用于matplotlib"""
        if hasattr(data, 'get'):  # CuPy数组
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data

    def train_single_model(self, aug_name, aug_config):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - Augmentation: {aug_config['name']}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        step_losses = []  # 记录每个step的loss
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
                if aug_config['function'] is not None:
                    # 重塑为4D数组以应用数据增强
                    x = x.reshape(-1, 3, 32, 32)
                    x = aug_config['function'](x.copy())
                    # 重新扁平化
                    x = x.reshape(x.shape[0], -1)
                
                model.zero_grad()
                y_pred = model.forward(x, training=False)  # 不使用dropout
          
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=0.0)  # 不使用L2正则化
                step_losses.append(loss)  # 记录每个step的loss
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                # 更新参数
                optimizer.step(model)
            
            # 验证准确率
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(float(val_acc))
            
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 测试准确率
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        # 训练集准确率
        train_pred = model.forward(train_images, training=False)
        train_acc = np.mean(np.argmax(train_pred, axis=1) == np.argmax(train_labels, axis=1))
        
        print(f"✅ 训练完成 - 训练准确率: {train_acc:.4f}, 最终测试准确率: {test_acc:.4f}")
        
        return {
            'augmentation_method': aug_name,
            'augmentation_name': aug_config['name'],
            'step_losses': [float(x) for x in self._to_numpy(step_losses)],
            'val_accuracies': [float(x) for x in val_accuracies],
            'test_accuracy': float(test_acc),
            'train_accuracy': float(train_acc),
            'best_val_accuracy': float(max(val_accuracies)),
            'steps_per_epoch': len(train_images) // self.batch_size
        }
    
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
        print("="*80)
        
        # 第一组：单独测试每种方法
        print(f"\n第一组实验：单独测试每种数据增强方法")
        print(f"测试方法: {', '.join([cfg['name'] for cfg in self.augmentation_configs_single.values()])}")
        print("-"*80)
        
        for aug_name, aug_config in self.augmentation_configs_single.items():
            result = self.train_single_model(aug_name, aug_config)
            self.results_single[aug_name] = result
        
        # 第二组：叠加测试
        print(f"\n第二组实验：叠加测试数据增强方法")
        print(f"测试方法: {', '.join([cfg['name'] for cfg in self.augmentation_configs_combined.values()])}")
        print("-"*80)
        
        for aug_name, aug_config in self.augmentation_configs_combined.items():
            result = self.train_single_model(aug_name, aug_config)
            self.results_combined[aug_name] = result
        
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
                'dropout': 0.0
            },
            'results_single': self.results_single,
            'results_combined': self.results_combined
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 定义颜色方案
        colors_single = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        colors_combined = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # ========== 第一组图表：单独测试每种方法 ==========
        fig1 = plt.figure(figsize=(18, 5))
        
        # 1.1 训练Loss曲线（每500步采样一次）
        ax1 = plt.subplot(1, 3, 1)
        sample_interval = 500
        for idx, (aug_name, result) in enumerate(self.results_single.items()):
            step_losses_np = np.array(result['step_losses'])
            sampled_steps = range(0, len(step_losses_np), sample_interval)
            sampled_losses = step_losses_np[sampled_steps]
            
            ax1.plot(sampled_steps, sampled_losses, 
                    label=result['augmentation_name'],
                    color=colors_single[idx % len(colors_single)],
                    linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Training Steps (sampled every 500 steps)', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Training Loss - Single Methods', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(left=0)
        
        # 1.2 验证准确率曲线
        ax2 = plt.subplot(1, 3, 2)
        for idx, (aug_name, result) in enumerate(self.results_single.items()):
            ax2.plot(range(1, self.epochs + 1), 
                    result['val_accuracies'],
                    label=result['augmentation_name'],
                    color=colors_single[idx % len(colors_single)],
                    linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Validation Accuracy', fontsize=11)
        ax2.set_title('Validation Accuracy - Single Methods', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 1.3 训练集和测试集准确率柱状图
        ax3 = plt.subplot(1, 3, 3)
        
        aug_names = [result['augmentation_name'] for result in self.results_single.values()]
        train_accs = [result['train_accuracy'] for result in self.results_single.values()]
        test_accs = [result['test_accuracy'] for result in self.results_single.values()]
        
        x = self._to_numpy(np.arange(len(aug_names)))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, train_accs, width, 
                       label='Train Accuracy',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax3.bar(x + width/2, test_accs, width,
                       label='Test Accuracy',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=8)
        
        ax3.set_xlabel('Data Augmentation Method', fontsize=11)
        ax3.set_ylabel('Accuracy', fontsize=11)
        ax3.set_title('Train vs Test Accuracy - Single Methods', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(aug_names, rotation=15, ha='right')
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        plot_filename1 = f"data_augmentation_single_plots_{timestamp}.png"
        plt.savefig(plot_filename1, bbox_inches='tight')
        print(f"📊 第一组图表已保存到: {plot_filename1}")
        plt.close()
        
        # ========== 第二组图表：叠加测试 ==========
        fig2 = plt.figure(figsize=(18, 5))
        
        # 2.1 训练Loss曲线（每500步采样一次）
        ax4 = plt.subplot(1, 3, 1)
        sample_interval = 500
        for idx, (aug_name, result) in enumerate(self.results_combined.items()):
            step_losses_np = np.array(result['step_losses'])
            sampled_steps = range(0, len(step_losses_np), sample_interval)
            sampled_losses = step_losses_np[sampled_steps]
            
            ax4.plot(sampled_steps, sampled_losses, 
                    label=result['augmentation_name'],
                    color=colors_combined[idx % len(colors_combined)],
                    linewidth=2, alpha=0.8)
        
        ax4.set_xlabel('Training Steps (sampled every 500 steps)', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11)
        ax4.set_title('Training Loss - Combined Methods', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_xlim(left=0)
        
        # 2.2 验证准确率曲线
        ax5 = plt.subplot(1, 3, 2)
        for idx, (aug_name, result) in enumerate(self.results_combined.items()):
            ax5.plot(range(1, self.epochs + 1), 
                    result['val_accuracies'],
                    label=result['augmentation_name'],
                    color=colors_combined[idx % len(colors_combined)],
                    linewidth=2, alpha=0.8)
        
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Validation Accuracy', fontsize=11)
        ax5.set_title('Validation Accuracy - Combined Methods', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 2.3 验证集和测试集准确率柱状图
        ax6 = plt.subplot(1, 3, 3)
        
        aug_names_combined = [result['augmentation_name'] for result in self.results_combined.values()]
        val_accs_combined = [result['best_val_accuracy'] for result in self.results_combined.values()]
        test_accs_combined = [result['test_accuracy'] for result in self.results_combined.values()]
        
        x_combined = self._to_numpy(np.arange(len(aug_names_combined)))
        width = 0.35
        
        bars3 = ax6.bar(x_combined - width/2, val_accs_combined, width, 
                       label='Validation Accuracy',
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars4 = ax6.bar(x_combined + width/2, test_accs_combined, width,
                       label='Test Accuracy',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        # 添加数值标签
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=8)
        
        ax6.set_xlabel('Data Augmentation Method', fontsize=11)
        ax6.set_ylabel('Accuracy', fontsize=11)
        ax6.set_title('Val vs Test Accuracy - Combined Methods', fontsize=12, fontweight='bold')
        ax6.set_xticks(x_combined)
        ax6.set_xticklabels(aug_names_combined, rotation=15, ha='right')
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax6.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        plot_filename2 = f"data_augmentation_combined_plots_{timestamp}.png"
        plt.savefig(plot_filename2, bbox_inches='tight')
        print(f"📊 第二组图表已保存到: {plot_filename2}")
        plt.close()
    
    def _print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*80}")
        print("实验总结")
        print(f"{'='*80}")
        
        print("\n第一组结果 (单独测试):")
        for aug_name, result in self.results_single.items():
            print(f"  {result['augmentation_name']:20s}: Train Acc={result['train_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        print("\n第二组结果 (叠加测试):")
        for aug_name, result in self.results_combined.items():
            print(f"  {result['augmentation_name']:25s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        # 找出单独测试中最佳的方法
        best_single = max(self.results_single.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 第一组最佳方法 (基于测试集准确率): {best_single[1]['augmentation_name']}")
        print(f"   训练准确率: {best_single[1]['train_accuracy']:.4f}")
        print(f"   测试准确率: {best_single[1]['test_accuracy']:.4f}")
        
        # 找出叠加测试中最佳的方法
        best_combined = max(self.results_combined.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 第二组最佳方法 (基于测试集准确率): {best_combined[1]['augmentation_name']}")
        print(f"   验证准确率: {best_combined[1]['best_val_accuracy']:.4f}")
        print(f"   测试准确率: {best_combined[1]['test_accuracy']:.4f}")
        
        # 分析叠加是否带来增益
        flip_only = self.results_combined['Flip_Only']['test_accuracy']
        flip_crop = self.results_combined['Flip_Crop']['test_accuracy']
        flip_crop_rotate = self.results_combined['Flip_Crop_Rotate']['test_accuracy']
        flip_crop_rotate_noise = self.results_combined['Flip_Crop_Rotate_Noise']['test_accuracy']
        
        print(f"\n📈 叠加增益分析:")
        print(f"   Flip Only:                    {flip_only:.4f}")
        print(f"   Flip + Crop:                  {flip_crop:.4f} (增益: {flip_crop - flip_only:+.4f})")
        print(f"   Flip + Crop + Rotate:         {flip_crop_rotate:.4f} (增益: {flip_crop_rotate - flip_crop:+.4f})")
        print(f"   Flip + Crop + Rotate + Noise: {flip_crop_rotate_noise:.4f} (增益: {flip_crop_rotate_noise - flip_crop_rotate:+.4f})")
        
        print(f"{'='*80}\n")

def main():
    experiment = DataAugmentationExperiment()
    experiment.run_experiments()

if __name__ == "__main__":
    main()

