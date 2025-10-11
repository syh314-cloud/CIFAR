"""
Dropout正则化对比实验
对比不同Dropout配置对模型性能的影响
生成两组实验的柱状图：验证集和测试集准确率对比
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
import os
from datetime import datetime

# 设置中文字体和显示参数
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

from utils.data_loader import train_images, train_labels, val_images, val_labels, test_images, test_labels
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import Adam

class DropoutExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 基准配置（与L2正则化实验相同）
        self.learning_rate = 7e-5
        self.batch_size = 64
        self.epochs = 100
        
        # 两组Dropout配置
        # 第一组：前10个epoch关闭Dropout
        self.group1_configs = {
            'p=(0.1,0.2,0.3)': {'warmup': 10, 'p': (0.1, 0.2, 0.3)},
            'p=(0.2,0.3,0.4)': {'warmup': 10, 'p': (0.2, 0.3, 0.4)},
            'p=(0.3,0.4,0.5)': {'warmup': 10, 'p': (0.3, 0.4, 0.5)},
            'p=(0.2,0.3,0.3)': {'warmup': 10, 'p': (0.2, 0.3, 0.3)},
            'p=(0.2,0.3,0.2)': {'warmup': 10, 'p': (0.2, 0.3, 0.2)},
        }
        
        # 第二组：前20个epoch关闭Dropout
        self.group2_configs = {
            'p=(0.1,0.2,0.3)': {'warmup': 20, 'p': (0.1, 0.2, 0.3)},
            'p=(0.2,0.3,0.4)': {'warmup': 20, 'p': (0.2, 0.3, 0.4)},
            'p=(0.3,0.4,0.5)': {'warmup': 20, 'p': (0.3, 0.4, 0.5)},
            'p=(0.2,0.3,0.3)': {'warmup': 20, 'p': (0.2, 0.3, 0.3)},
            'p=(0.2,0.3,0.2)': {'warmup': 20, 'p': (0.2, 0.3, 0.2)},
        }
        
        self.results_group1 = {}
        self.results_group2 = {}
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组用于matplotlib"""
        if hasattr(data, 'get'):
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data

    def train_single_model(self, config_name, config, group_name):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - {group_name} - {config_name}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型（与L2实验相同的结构）
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        warmup_epochs = config['warmup']
        dropout_p = config['p']
        
        step_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        
        for epoch in range(self.epochs):
            # 设置dropout策略
            if epoch < warmup_epochs:
                # Warmup阶段：关闭Dropout
                model.dropout1.p = 0.0
                model.dropout2.p = 0.0
                model.dropout3.p = 0.0
            else:
                # 正式训练：使用指定的Dropout概率
                model.dropout1.p = dropout_p[0]
                model.dropout2.p = dropout_p[1]
                model.dropout3.p = dropout_p[2]
            
            np.random.seed(self.SEED + epoch)
            idx = np.random.permutation(train_images.shape[0])
            shuffled_images = train_images[idx]
            shuffled_labels = train_labels[idx]
            
            epoch_losses = []
            
            for i in range(0, shuffled_images.shape[0], self.batch_size):
                x = shuffled_images[i:i+self.batch_size]
                y = shuffled_labels[i:i+self.batch_size]
                
                model.zero_grad()
                y_pred = model.forward(x, training=True)
          
                # 不使用L2正则化
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=0.0)
                step_losses.append(loss)
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                optimizer.step(model)
            
            # 验证
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(val_acc)
            
            # 转换为标量
            if hasattr(val_acc, 'get'):
                val_acc_scalar = float(val_acc.get())
            else:
                val_acc_scalar = float(val_acc)
            
            if val_acc_scalar > best_val_acc:
                best_val_acc = val_acc_scalar
            
            avg_loss = np.mean(np.array(epoch_losses))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc_scalar:.4f}, Best Val: {best_val_acc:.4f}")
        
        # 测试
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        # 转换为标量
        if hasattr(test_acc, 'get'):
            test_acc_scalar = float(test_acc.get())
        else:
            test_acc_scalar = float(test_acc)
        
        print(f"✅ 训练完成 - 最佳验证准确率: {best_val_acc:.4f}, 测试准确率: {test_acc_scalar:.4f}")
        
        return {
            'step_losses': self._to_numpy(step_losses),
            'val_accuracies': self._to_numpy(val_accuracies),
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc_scalar,
            'warmup_epochs': warmup_epochs,
            'dropout_p': dropout_p,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n{'='*80}")
        print(f"开始 Dropout 正则化对比实验")
        print(f"{'='*80}")
        print(f"配置信息:")
        print(f"  - 学习率: {self.learning_rate}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - ❌ 已关闭L2正则化和数据增强")
        print(f"{'='*80}\n")
        
        # 第一组实验：前10个epoch关闭Dropout
        print(f"\n{'='*80}")
        print(f"第一组实验：前 10 个 epoch 关闭 Dropout")
        print(f"{'='*80}")
        for config_name, config in self.group1_configs.items():
            result = self.train_single_model(config_name, config, "Group1")
            self.results_group1[config_name] = result
        
        # 第二组实验：前20个epoch关闭Dropout
        print(f"\n{'='*80}")
        print(f"第二组实验：前 20 个 epoch 关闭 Dropout")
        print(f"{'='*80}")
        for config_name, config in self.group2_configs.items():
            result = self.train_single_model(config_name, config, "Group2")
            self.results_group2[config_name] = result
        
        # 保存结果
        self.save_results()
        
        # 绘制图表
        self.plot_results()
    
    def save_results(self):
        """保存实验结果到JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'dropout_comparison_results_{timestamp}.json'
        
        # 预处理结果，确保所有数据都是可序列化的
        def process_results(results_dict):
            processed = {}
            for config_name, result in results_dict.items():
                processed[config_name] = {
                    'step_losses': [float(x) if hasattr(x, 'get') else float(x) for x in result['step_losses']],
                    'val_accuracies': [float(x) if hasattr(x, 'get') else float(x) for x in result['val_accuracies']],
                    'best_val_accuracy': float(result['best_val_accuracy']),
                    'test_accuracy': float(result['test_accuracy']),
                    'warmup_epochs': result['warmup_epochs'],
                    'dropout_p': result['dropout_p'],
                    'learning_rate': float(result['learning_rate']),
                    'batch_size': int(result['batch_size'])
                }
            return processed
        
        results = {
            'group1': process_results(self.results_group1),
            'group2': process_results(self.results_group2)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        print(f"\n📊 生成可视化图表...")
        
        # 创建两个子图
        plt.close('all')
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Dropout Regularization Comparison: Validation & Test Accuracy', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 绘制第一组结果
        self._plot_group(axes[0], self.results_group1, 
                        'Group 1: Warmup 10 Epochs (Dropout OFF) → Dropout ON',
                        'Warmup=10')
        
        # 绘制第二组结果
        self._plot_group(axes[1], self.results_group2, 
                        'Group 2: Warmup 20 Epochs (Dropout OFF) → Dropout ON',
                        'Warmup=20')
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'dropout_comparison_plots_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"✅ 图表已保存到: {filename}")
        
        # 打印结果总结
        self._print_summary()
    
    def _plot_group(self, ax, results, title, group_label):
        """绘制单个组的柱状图"""
        config_names = list(results.keys())
        val_accs = [results[name]['best_val_accuracy'] for name in config_names]
        test_accs = [results[name]['test_accuracy'] for name in config_names]
        
        # 设置柱状图参数
        x = self._to_numpy(np.arange(len(config_names)))
        width = 0.35
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, val_accs, width, label='Validation Accuracy',
                      color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
                      color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加标签和标题
        ax.set_xlabel('Dropout Configuration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(config_names, fontsize=10, rotation=15, ha='right')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=8)
        
        # 设置y轴范围
        y_max = max(max(val_accs), max(test_accs))
        ax.set_ylim(0, y_max * 1.15)
    
    def _print_summary(self):
        """打印结果总结"""
        print(f"\n{'='*80}")
        print(f"实验结果总结")
        print(f"{'='*80}")
        
        print(f"\n第一组 (Warmup 10 Epochs):")
        for name, result in self.results_group1.items():
            print(f"  {name:20s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        print(f"\n第二组 (Warmup 20 Epochs):")
        for name, result in self.results_group2.items():
            print(f"  {name:20s}: Val Acc={result['best_val_accuracy']:.4f}, Test Acc={result['test_accuracy']:.4f}")
        
        # 找出最佳配置
        all_results = {**self.results_group1, **self.results_group2}
        best_config = max(all_results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 最佳配置: {best_config[0]}")
        print(f"   验证准确率: {best_config[1]['best_val_accuracy']:.4f}")
        print(f"   测试准确率: {best_config[1]['test_accuracy']:.4f}")
        print(f"   Warmup Epochs: {best_config[1]['warmup_epochs']}")
        print(f"   Dropout p: {best_config[1]['dropout_p']}")
        print(f"{'='*80}\n")

def main():
    experiment = DropoutExperiment()
    experiment.run_experiments()

if __name__ == '__main__':
    main()

