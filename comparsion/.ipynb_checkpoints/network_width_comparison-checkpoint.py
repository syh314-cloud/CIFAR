"""
MLP网络宽度对比实验
批量训练模型，对比不同隐藏层神经元数量的效果
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
from model.mlp_2layer import MLP
from utils.loss import CrossEntropyLoss, L2Scheduler
from utils.classic_optimizers import Adam
from utils.data_augmentation import augment_images

class NetworkWidthExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        
        # 要测试的隐藏层神经元数量
        self.hidden_dims = [128, 256, 512, 1024, 2048]
        
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

    def train_single_model(self, hidden_dim):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - Hidden Dim: {hidden_dim}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型 - 使用2层MLP
        model = MLP(train_images.shape[1], hidden_dim, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        l2_scheduler = L2Scheduler(base_lambda=1e-4)
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        lambda_l2 = l2_scheduler.base_lambda
        
        step_losses = []  # 记录每个step的loss
        val_accuracies = []
        steps_per_epoch = len(range(0, train_images.shape[0], self.batch_size))
        
        for epoch in range(self.epochs):
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
                y_pred = model.forward(x)
          
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
                step_losses.append(loss)  # 记录每个step的loss
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                for layer in model.layers:
                    if hasattr(layer, 'w'):
                        layer.dw += lambda_l2 * layer.w
                
                optimizer.step(model)
            
            val_pred = model.forward(val_images)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(val_acc)
            
            avg_loss = np.mean(np.array(epoch_losses))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        test_pred = model.forward(test_images)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        print(f"✅ 训练完成 - 最终测试准确率: {test_acc:.4f}")
        
        # 计算模型参数量
        total_params = 0
        for layer in model.layers:
            if hasattr(layer, 'w'):
                total_params += layer.w.size + layer.b.size
        
        return {
            'step_losses': step_losses,  # 每个step的loss
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'hidden_dim': hidden_dim,
            'steps_per_epoch': steps_per_epoch,
            'total_params': int(total_params)
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 开始MLP网络宽度对比实验...")
        print(f"📊 固定学习率: {self.learning_rate}")
        print(f"📦 固定Batch Size: {self.batch_size}")
        print(f"🏗️  测试隐藏层维度: {self.hidden_dims}")
        print("-" * 60)
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            print(f"\n{'='*60}")
            print(f"实验进度: {i+1}/{len(self.hidden_dims)}")
            
            result = self.train_single_model(hidden_dim)
            self.results[hidden_dim] = result
        
        print(f"\n🎉 所有实验完成！")
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_width_comparison_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {}
        for hidden_dim, result in self.results.items():
            step_losses = [float(x.get() if hasattr(x, 'get') else x) for x in result['step_losses']]
            val_accuracies = [float(x.get() if hasattr(x, 'get') else x) for x in result['val_accuracies']]
            test_accuracy = float(result['test_accuracy'].get() if hasattr(result['test_accuracy'], 'get') else result['test_accuracy'])
            results_to_save[str(hidden_dim)] = {
                'step_losses': step_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_accuracy,
                'hidden_dim': result['hidden_dim'],
                'steps_per_epoch': result['steps_per_epoch'],
                'total_params': result['total_params'],
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
        fig.suptitle('Network Width Comparison Results (2-Layer MLP)', fontsize=18, fontweight='bold', y=0.98)
        
        # 定义颜色方案
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # 1. 训练Loss曲线 (按step)
        ax1 = axes[0, 0]
        for idx, (hidden_dim, result) in enumerate(self.results.items()):
            step_losses_np = self._to_numpy(result['step_losses'])
            steps = range(len(step_losses_np))
            # 每隔一定步数采样，避免图表过于密集
            sample_interval = 100
            sampled_steps = steps[::sample_interval]
            sampled_losses = step_losses_np[::sample_interval]
            ax1.plot(sampled_steps, sampled_losses, label=f'Hidden={hidden_dim}', 
                    color=colors[idx], linewidth=1.5, alpha=0.8)
        ax1.set_title('Training Loss Curves (Sampled every 100 steps)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_xlim(left=0)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # 2. 验证Accuracy曲线
        ax2 = axes[0, 1]
        for idx, (hidden_dim, result) in enumerate(self.results.items()):
            val_accuracies_np = self._to_numpy(result['val_accuracies'])
            ax2.plot(val_accuracies_np, label=f'Hidden={hidden_dim}', 
                    color=colors[idx], linewidth=2, marker='s', markersize=3)
        ax2.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # 3. 测试准确率柱状图
        ax3 = axes[1, 0]
        hidden_dims = list(self.results.keys())
        test_accs = [self._to_numpy(self.results[hd]['test_accuracy']) for hd in hidden_dims]
        
        bars = ax3.bar(range(len(hidden_dims)), test_accs, 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Hidden Layer Dimension', fontsize=12)
        ax3.set_ylabel('Test Accuracy', fontsize=12)
        ax3.set_xticks(range(len(hidden_dims)))
        ax3.set_xticklabels([f'{hd}' for hd in hidden_dims], fontsize=10)
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
        headers = ['Hidden Dim', '参数量', '最终Loss', '最佳Val Acc', '测试准确率']
        
        for hidden_dim in hidden_dims:
            result = self.results[hidden_dim]
            total_params = result['total_params']
            final_loss = self._to_numpy(result['step_losses'][-1])
            best_val_acc = max(self._to_numpy(result['val_accuracies']))
            test_acc = self._to_numpy(result['test_accuracy'])
            
            table_data.append([
                f'{hidden_dim}',
                f'{total_params:,}',
                f'{final_loss:.4f}',
                f'{best_val_acc:.4f}',
                f'{test_acc:.4f}'
            ])
        
        # 创建表格
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.15, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
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
        filename = f'network_width_comparison_plots_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📈 图表已保存到: {filename}")
        
        # 尝试显示图表（如果有GUI环境）
        try:
            plt.show()
        except:
            print("📱 无GUI环境，图表已保存为文件")
        
        # 打印最佳结果
        best_hd = max(self.results.keys(), key=lambda hd: self._to_numpy(self.results[hd]['test_accuracy']))
        best_acc = self._to_numpy(self.results[best_hd]['test_accuracy'])
        print(f"\n🏆 最佳网络宽度: {best_hd} (测试准确率: {best_acc:.4f})")

def main():
    """主函数"""
    print("🎯 MLP网络宽度对比实验")
    print("=" * 60)
    
    # 创建实验对象
    experiment = NetworkWidthExperiment()
    
    # 运行实验
    experiment.run_experiments()
    
    print("\n✅ 实验完成！")
    print("📊 已生成:")
    print("  - 训练Loss曲线 (按步数)")
    print("  - 验证Accuracy曲线") 
    print("  - 测试准确率柱状图")
    print("  - 性能对比表格（包含参数量）")

if __name__ == "__main__":
    main()
