"""
损失函数对比实验
批量训练模型，对比不同损失函数的效果
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

from utils.data_loader import train_images, train_labels, val_images, val_labels, test_images, test_labels
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss, LabelSmoothingLoss, FocalLoss
from utils.classic_optimizers import Adam
from utils.data_augmentation import augment_images

class LossFunctionExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        
        # 只测试FocalLoss
        self.loss_configs = {
            'FocalLoss(γ=2.0)': {
                'class': FocalLoss,
                'params': {'gamma': 2.0},
                'use_custom_backward': True
            }
        }
        
        self.results = {}
        self.epochs = 100
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组用于matplotlib"""
        if hasattr(data, 'get'):  # CuPy数组
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data

    def train_single_model(self, loss_name, loss_config):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - Loss Function: {loss_name}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = loss_config['class'](**loss_config['params'])
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        # 固定L2正则化系数
        lambda_l2 = 1e-4
        
        step_losses = []  # 记录每个step的loss
        val_accuracies = []
        steps_per_epoch = len(range(0, train_images.shape[0], self.batch_size))
        
        for epoch in range(self.epochs):
            # 设置dropout策略
            if epoch < 20:
                model.dropout1.p = 0.0
                model.dropout2.p = 0.0
                model.dropout3.p = 0.0
            else:
                model.dropout1.p = 0.2
                model.dropout2.p = 0.3
                model.dropout3.p = 0.3
            
            np.random.seed(self.SEED + epoch)
            idx = np.random.permutation(train_images.shape[0])
            shuffled_images = train_images[idx]
            shuffled_labels = train_labels[idx]
            
            epoch_losses = []
            
            for i in range(0, shuffled_images.shape[0], self.batch_size):
                x = shuffled_images[i:i+self.batch_size]
                y = shuffled_labels[i:i+self.batch_size]
                
                x = x.reshape(-1, 3, 32, 32)
                x = augment_images(x, seed=self.SEED + epoch * 1000 + i)
                x = x.reshape(x.shape[0], -1)
                
                model.zero_grad()
                y_pred = model.forward(x, training=True)
          
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
                step_losses.append(loss)  # 记录每个step的loss
                epoch_losses.append(loss)
                
                # 根据损失函数类型选择backward方法
                if loss_config['use_custom_backward']:
                    grad_output = loss_fn.backward(y_pred, y)
                else:
                    grad_output = loss_fn.backward()
                
                model.backward(grad_output)
                
                # 添加L2正则化梯度
                for layer in model.layers:
                    if hasattr(layer, 'w'):
                        layer.dw += lambda_l2 * layer.w
                
                optimizer.step(model)
            
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(val_acc)
            
            avg_loss = np.mean(np.array(epoch_losses))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        print(f"✅ 训练完成 - 最终测试准确率: {test_acc:.4f}")
        
        return {
            'step_losses': step_losses,  # 每个step的loss
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'loss_name': loss_name,
            'steps_per_epoch': steps_per_epoch
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 开始损失函数对比实验...")
        print(f"📊 固定学习率: {self.learning_rate}")
        print(f"📦 固定Batch Size: {self.batch_size}")
        print(f"🔧 测试损失函数: {list(self.loss_configs.keys())}")
        print("-" * 60)
        
        for i, (loss_name, loss_config) in enumerate(self.loss_configs.items()):
            print(f"\n{'='*60}")
            print(f"实验进度: {i+1}/{len(self.loss_configs)}")
            
            result = self.train_single_model(loss_name, loss_config)
            self.results[loss_name] = result
        
        print(f"\n🎉 所有实验完成！")
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"loss_function_comparison_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {}
        for loss_name, result in self.results.items():
            step_losses = [float(x.get() if hasattr(x, 'get') else x) for x in result['step_losses']]
            val_accuracies = [float(x.get() if hasattr(x, 'get') else x) for x in result['val_accuracies']]
            test_accuracy = float(result['test_accuracy'].get() if hasattr(result['test_accuracy'], 'get') else result['test_accuracy'])
            results_to_save[loss_name] = {
                'step_losses': step_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_accuracy,
                'loss_name': result['loss_name'],
                'steps_per_epoch': result['steps_per_epoch'],
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"📁 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        print(f"\n📊 生成可视化图表...")
        
        # 创建图表
        plt.close('all')  # 关闭之前的图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FocalLoss Training Results (γ=2.0)', fontsize=18, fontweight='bold', y=0.98)
        
        # 定义颜色方案
        colors = {
            'FocalLoss(γ=2.0)': '#FF6B6B'  # 红色
        }
        
        # 1. 训练Loss曲线 (按step)
        ax1 = axes[0, 0]
        for loss_name, result in self.results.items():
            step_losses_np = self._to_numpy(result['step_losses'])
            steps = range(len(step_losses_np))
            # 每隔一定步数采样，避免图表过于密集
            sample_interval = 500
            sampled_steps = steps[::sample_interval]
            sampled_losses = step_losses_np[::sample_interval]
            ax1.plot(sampled_steps, sampled_losses, label=loss_name, 
                    color=colors.get(loss_name, 'gray'), linewidth=1.5, alpha=0.8)
        ax1.set_title('Training Loss Curves (Sampled every 500 steps)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_xlim(left=0)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # 2. 验证Accuracy曲线
        ax2 = axes[0, 1]
        for loss_name, result in self.results.items():
            val_accuracies_np = self._to_numpy(result['val_accuracies'])
            ax2.plot(val_accuracies_np, label=loss_name, 
                    color=colors.get(loss_name, 'gray'), linewidth=2, marker='s', markersize=3)
        ax2.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # 3. 测试准确率柱状图
        ax3 = axes[1, 0]
        loss_names = list(self.results.keys())
        test_accs = [self._to_numpy(self.results[name]['test_accuracy']) for name in loss_names]
        
        bar_colors = [colors.get(name, 'gray') for name in loss_names]
        bars = ax3.bar(range(len(loss_names)), test_accs, 
                      color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Loss Function', fontsize=12)
        ax3.set_ylabel('Test Accuracy', fontsize=12)
        ax3.set_xticks(range(len(loss_names)))
        ax3.set_xticklabels(loss_names, fontsize=9, rotation=15, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='both', which='major', labelsize=10)
        
        # 在柱状图上添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, test_accs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. 最终收敛性能对比表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['Loss Function', '最终Loss', '最佳Val Acc', '测试准确率', 'Steps/Epoch']
        
        for loss_name in loss_names:
            result = self.results[loss_name]
            final_loss = self._to_numpy(result['step_losses'][-1])
            best_val_acc = max(self._to_numpy(result['val_accuracies']))
            test_acc = self._to_numpy(result['test_accuracy'])
            steps_per_epoch = result['steps_per_epoch']
            
            table_data.append([
                loss_name,
                f'{final_loss:.4f}',
                f'{best_val_acc:.4f}',
                f'{test_acc:.4f}',
                f'{steps_per_epoch}'
            ])
        
        # 创建表格
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'loss_function_comparison_plots_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📈 图表已保存到: {filename}")
        
        # 尝试显示图表（如果有GUI环境）
        try:
            plt.show()
        except:
            print("📱 无GUI环境，图表已保存为文件")
        
        # 打印最佳结果
        best_loss = max(self.results.keys(), key=lambda name: self._to_numpy(self.results[name]['test_accuracy']))
        best_acc = self._to_numpy(self.results[best_loss]['test_accuracy'])
        print(f"\n🏆 最佳损失函数: {best_loss} (测试准确率: {best_acc:.4f})")

def main():
    """主函数"""
    print("🎯 损失函数对比实验")
    print("=" * 60)
    
    # 创建实验对象
    experiment = LossFunctionExperiment()
    
    # 运行实验
    experiment.run_experiments()
    
    print("\n✅ 实验完成！")
    print("📊 已生成:")
    print("  - 训练Loss曲线 (按步数)")
    print("  - 验证Accuracy曲线") 
    print("  - 测试准确率柱状图")
    print("  - 性能对比表格")

if __name__ == "__main__":
    main()
