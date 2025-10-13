"""
正则化组合对比实验
固定参数：
- L2正则化系数：5e-4
- Dropout：Warmup=10, p=(0.2, 0.3, 0.2)
- 数据增强：Crop + Flip

测试组合：
1. Baseline（无正则化）
2. L2 only
3. Dropout only
4. Data Augmentation only
5. L2 + Dropout
6. L2 + Data Augmentation
7. Dropout + Data Augmentation
8. L2 + Dropout + Data Augmentation（全部组合）

目标：找到最佳正则化配置
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
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
from utils.data_augmentation import random_flip, random_crop
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import Adam

class RegularizationCombinationExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        self.epochs = 100
        
        # 固定的正则化参数
        self.l2_lambda = 5e-4
        self.dropout_warmup = 10
        self.dropout_p = (0.2, 0.3, 0.2)  # (p1, p2, p3)
        
        # 定义所有测试配置
        self.configs = {
            'Baseline': {
                'name': 'Baseline',
                'use_l2': False,
                'use_dropout': False,
                'use_augmentation': False,
                'description': '无正则化'
            },
            'L2': {
                'name': 'L2 Only',
                'use_l2': True,
                'use_dropout': False,
                'use_augmentation': False,
                'description': f'L2正则化 (λ={self.l2_lambda})'
            },
            'Dropout': {
                'name': 'Dropout Only',
                'use_l2': False,
                'use_dropout': True,
                'use_augmentation': False,
                'description': f'Dropout (Warmup={self.dropout_warmup}, p={self.dropout_p})'
            },
            'Augmentation': {
                'name': 'Data Aug Only',
                'use_l2': False,
                'use_dropout': False,
                'use_augmentation': True,
                'description': '数据增强 (Crop + Flip)'
            },
            'L2_Dropout': {
                'name': 'L2 + Dropout',
                'use_l2': True,
                'use_dropout': True,
                'use_augmentation': False,
                'description': 'L2 + Dropout'
            },
            'L2_Augmentation': {
                'name': 'L2 + Data Aug',
                'use_l2': True,
                'use_dropout': False,
                'use_augmentation': True,
                'description': 'L2 + 数据增强'
            },
            'Dropout_Augmentation': {
                'name': 'Dropout + Data Aug',
                'use_l2': False,
                'use_dropout': True,
                'use_augmentation': True,
                'description': 'Dropout + 数据增强'
            },
            'All': {
                'name': 'L2 + Dropout + Data Aug',
                'use_l2': True,
                'use_dropout': True,
                'use_augmentation': True,
                'description': '全部正则化'
            }
        }
        
        self.results = {}
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组"""
        if hasattr(data, 'get'):
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data
    
    def _apply_augmentation(self, x):
        """应用Crop + Flip数据增强"""
        x = x.reshape(-1, 3, 32, 32)
        x = random_flip(x.copy(), prob=0.5)
        x = random_crop(x.copy(), crop_size=32, padding=4)
        return x.reshape(x.shape[0], -1)
    
    def train_single_model(self, config_name, config):
        """训练单个模型"""
        print(f"\n{'='*80}")
        print(f"🔧 开始训练 - {config['name']}")
        print(f"   配置: {config['description']}")
        print(f"{'='*80}")
        
        # 重置随机种子
        np.random.seed(self.SEED)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        step_losses = []
        val_accuracies = []
        
        # 配置Dropout
        if config['use_dropout']:
            # Dropout会在训练循环中根据epoch设置
            pass
        else:
            # 关闭Dropout
            model.dropout1.p = 0.0
            model.dropout2.p = 0.0
            model.dropout3.p = 0.0
        
        # 配置L2正则化
        lambda_l2 = self.l2_lambda if config['use_l2'] else 0.0
        
        for epoch in range(self.epochs):
            # 设置Dropout
            if config['use_dropout']:
                if epoch < self.dropout_warmup:
                    model.dropout1.p = 0.0
                    model.dropout2.p = 0.0
                    model.dropout3.p = 0.0
                else:
                    model.dropout1.p = self.dropout_p[0]
                    model.dropout2.p = self.dropout_p[1]
                    model.dropout3.p = self.dropout_p[2]
            
            np.random.seed(self.SEED + epoch)
            idx = np.random.permutation(train_images.shape[0])
            shuffled_images = train_images[idx]
            shuffled_labels = train_labels[idx]
            
            epoch_losses = []
            
            for i in range(0, shuffled_images.shape[0], self.batch_size):
                x = shuffled_images[i:i+self.batch_size]
                y = shuffled_labels[i:i+self.batch_size]
                
                # 应用数据增强
                if config['use_augmentation']:
                    x = self._apply_augmentation(x)
                
                model.zero_grad()
                
                # 根据是否使用Dropout决定training模式
                training_mode = config['use_dropout']
                y_pred = model.forward(x, training=training_mode)
                
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
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
        
        print(f"✅ 训练完成 - 最佳验证准确率: {max(val_accuracies):.4f}, 测试准确率: {test_acc:.4f}")
        
        return {
            'config_name': config_name,
            'display_name': config['name'],
            'description': config['description'],
            'use_l2': config['use_l2'],
            'use_dropout': config['use_dropout'],
            'use_augmentation': config['use_augmentation'],
            'step_losses': [float(x) for x in self._to_numpy(step_losses)],
            'val_accuracies': [float(x) for x in val_accuracies],
            'test_accuracy': float(test_acc),
            'best_val_accuracy': float(max(val_accuracies)),
            'steps_per_epoch': len(train_images) // self.batch_size
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print("="*80)
        print("正则化组合对比实验")
        print("="*80)
        print(f"固定参数:")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"\n正则化参数:")
        print(f"  - L2 Lambda: {self.l2_lambda}")
        print(f"  - Dropout Warmup: {self.dropout_warmup} epochs")
        print(f"  - Dropout p: {self.dropout_p}")
        print(f"  - Data Augmentation: Crop + Flip")
        print(f"\n测试配置数量: {len(self.configs)}")
        print("="*80)
        
        # 按顺序训练所有配置
        for config_name, config in self.configs.items():
            result = self.train_single_model(config_name, config)
            self.results[config_name] = result
        
        self.save_results()
        self.plot_results()
        self._print_summary()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"regularization_combination_results_{timestamp}.json"
        
        save_data = {
            'experiment_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'seed': self.SEED,
                'l2_lambda': self.l2_lambda,
                'dropout_warmup': self.dropout_warmup,
                'dropout_p': self.dropout_p
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig = plt.figure(figsize=(20, 10))
        
        # 定义配置顺序和分组
        config_order = ['Baseline', 'L2', 'Dropout', 'Augmentation', 
                       'L2_Dropout', 'L2_Augmentation', 'Dropout_Augmentation', 'All']
        
        colors = {
            'Baseline': '#95a5a6',
            'L2': '#3498db',
            'Dropout': '#e74c3c',
            'Augmentation': '#2ecc71',
            'L2_Dropout': '#9b59b6',
            'L2_Augmentation': '#f39c12',
            'Dropout_Augmentation': '#1abc9c',
            'All': '#e67e22'
        }
        
        # 1. 验证准确率曲线
        ax1 = plt.subplot(2, 3, 1)
        for config_name in config_order:
            if config_name in self.results:
                result = self.results[config_name]
                ax1.plot(range(1, self.epochs + 1),
                        result['val_accuracies'],
                        label=result['display_name'],
                        color=colors[config_name],
                        linewidth=2, alpha=0.8,
                        linestyle='--' if config_name == 'Baseline' else '-')
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Validation Accuracy', fontsize=11)
        ax1.set_title('Validation Accuracy Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9, fontsize=8)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. 验证集和测试集准确率对比（柱状图）
        ax2 = plt.subplot(2, 3, 2)
        
        display_names = [self.results[name]['display_name'] for name in config_order if name in self.results]
        val_accs = [self.results[name]['best_val_accuracy'] for name in config_order if name in self.results]
        test_accs = [self.results[name]['test_accuracy'] for name in config_order if name in self.results]
        
        x = self._to_numpy(np.arange(len(display_names)))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, val_accs, width,
                       label='Validation Accuracy',
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax2.bar(x + width/2, test_accs, width,
                       label='Test Accuracy',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=7)
        
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Val vs Test Accuracy', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=30, ha='right', fontsize=8)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_ylim([0, 1.0])
        
        # 3. 相对Baseline的提升
        ax3 = plt.subplot(2, 3, 3)
        
        baseline_test = self.results['Baseline']['test_accuracy']
        improvements = []
        improvement_names = []
        improvement_colors = []
        
        for config_name in config_order:
            if config_name != 'Baseline' and config_name in self.results:
                improvement = self.results[config_name]['test_accuracy'] - baseline_test
                improvements.append(improvement)
                improvement_names.append(self.results[config_name]['display_name'])
                improvement_colors.append(colors[config_name])
        
        x_imp = self._to_numpy(np.arange(len(improvement_names)))
        bars = ax3.bar(x_imp, improvements, color=improvement_colors, alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:+.4f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Test Accuracy Improvement', fontsize=11)
        ax3.set_title('Improvement over Baseline', fontsize=12, fontweight='bold')
        ax3.set_xticks(x_imp)
        ax3.set_xticklabels(improvement_names, rotation=30, ha='right', fontsize=8)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 4. 单独方法 vs 组合方法对比
        ax4 = plt.subplot(2, 3, 4)
        
        single_methods = ['L2', 'Dropout', 'Augmentation']
        single_accs = [self.results[name]['test_accuracy'] for name in single_methods]
        
        combined_methods = ['L2_Dropout', 'L2_Augmentation', 'Dropout_Augmentation', 'All']
        combined_accs = [self.results[name]['test_accuracy'] for name in combined_methods]
        
        x_single = self._to_numpy(np.arange(len(single_methods)))
        x_combined = self._to_numpy(np.arange(len(combined_methods)))
        
        bars1 = ax4.bar(x_single, single_accs, 
                       color=[colors[name] for name in single_methods],
                       alpha=0.8, edgecolor='black', linewidth=0.8, label='Single')
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax4.axhline(y=baseline_test, color='#95a5a6', linestyle='--', linewidth=2, label='Baseline')
        ax4.set_ylabel('Test Accuracy', fontsize=11)
        ax4.set_title('Single Regularization Methods', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_single)
        ax4.set_xticklabels([self.results[name]['display_name'] for name in single_methods], 
                           rotation=20, ha='right', fontsize=9)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_ylim([0, 1.0])
        
        # 5. 组合方法对比
        ax5 = plt.subplot(2, 3, 5)
        
        bars2 = ax5.bar(x_combined, combined_accs,
                       color=[colors[name] for name in combined_methods],
                       alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bar in bars2:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)
        
        ax5.axhline(y=baseline_test, color='#95a5a6', linestyle='--', linewidth=2, label='Baseline')
        ax5.set_ylabel('Test Accuracy', fontsize=11)
        ax5.set_title('Combined Regularization Methods', fontsize=12, fontweight='bold')
        ax5.set_xticks(x_combined)
        ax5.set_xticklabels([self.results[name]['display_name'] for name in combined_methods],
                           rotation=20, ha='right', fontsize=8)
        ax5.legend(loc='best', framealpha=0.9)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_ylim([0, 1.0])
        
        # 6. 排名对比（Top配置）
        ax6 = plt.subplot(2, 3, 6)
        
        sorted_results = sorted(
            [(name, result) for name, result in self.results.items()],
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        top_n = min(5, len(sorted_results))
        top_names = [result[1]['display_name'] for result in sorted_results[:top_n]]
        top_test = [result[1]['test_accuracy'] for result in sorted_results[:top_n]]
        top_val = [result[1]['best_val_accuracy'] for result in sorted_results[:top_n]]
        
        x_top = self._to_numpy(np.arange(top_n))
        width = 0.35
        
        bars1 = ax6.bar(x_top - width/2, top_val, width,
                       label='Validation Acc',
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax6.bar(x_top + width/2, top_test, width,
                       label='Test Acc',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=8)
        
        ax6.set_ylabel('Accuracy', fontsize=11)
        ax6.set_title(f'Top {top_n} Configurations', fontsize=12, fontweight='bold')
        ax6.set_xticks(x_top)
        ax6.set_xticklabels(top_names, rotation=20, ha='right', fontsize=9)
        ax6.legend(loc='best', framealpha=0.9)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax6.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        plot_filename = f"regularization_combination_plots_{timestamp}.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"📊 图表已保存到: {plot_filename}")
        plt.close()
    
    def _print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*80}")
        print("实验总结")
        print(f"{'='*80}")
        
        # 按测试集准确率排序
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        print("\n📊 所有配置结果（按测试集准确率排序）：")
        print("-"*80)
        for rank, (config_name, result) in enumerate(sorted_results, 1):
            print(f"{rank}. {result['display_name']:25s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        # 最佳配置
        best_config_name, best_result = sorted_results[0]
        baseline_test = self.results['Baseline']['test_accuracy']
        improvement = best_result['test_accuracy'] - baseline_test
        
        print(f"\n🏆 最佳配置:")
        print(f"   配置: {best_result['display_name']}")
        print(f"   描述: {best_result['description']}")
        print(f"   验证准确率: {best_result['best_val_accuracy']:.4f}")
        print(f"   测试准确率: {best_result['test_accuracy']:.4f}")
        print(f"   相对Baseline提升: {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # 分析泛化性能
        gap = abs(best_result['best_val_accuracy'] - best_result['test_accuracy'])
        if gap > 0.02:
            print(f"   ⚠️ 验证集和测试集准确率差距: {gap:.4f}，可能存在过拟合")
        else:
            print(f"   ✅ 验证集和测试集准确率接近，泛化性能良好")
        
        # 单独方法对比
        print(f"\n📋 单独正则化方法效果:")
        for method in ['L2', 'Dropout', 'Augmentation']:
            result = self.results[method]
            improvement = result['test_accuracy'] - baseline_test
            print(f"   {result['display_name']:20s}: {result['test_accuracy']:.4f} ({improvement:+.4f})")
        
        # 组合方法对比
        print(f"\n📋 组合正则化方法效果:")
        for method in ['L2_Dropout', 'L2_Augmentation', 'Dropout_Augmentation', 'All']:
            result = self.results[method]
            improvement = result['test_accuracy'] - baseline_test
            print(f"   {result['display_name']:25s}: {result['test_accuracy']:.4f} ({improvement:+.4f})")
        
        print(f"{'='*80}\n")

def main():
    experiment = RegularizationCombinationExperiment()
    experiment.run_experiments()

if __name__ == "__main__":
    main()

