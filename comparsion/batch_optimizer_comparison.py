"""
Batch Optimizer对比实验
批量训练模型，对比不同批次优化器的效果
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
from collections import OrderedDict

# 设置中文字体和显示参数
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

from utils.data_loader import train_images, train_labels, val_images, val_labels, test_images, test_labels
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss, L2Scheduler
from utils.batch_optimizers import BatchGD, OnlineGD, MiniBatchGD, AdaptiveBatchGD
from utils.data_augmentation import augment_images

class BatchOptimizerExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 基准batch size
        self.batch_size = 64
        
        # 为每个优化器设置合理的学习率，让它们都在"合理工作点"
        # MiniBatchGD (batch=64)：7e-5 - 第一个训练
        # BatchGD (full batch)：3e-4 - 第二个训练
        # AdaptiveBatchGD：1e-4 - 第三个训练
        # OnlineGD (batch=1)：3e-4 - 第四个训练
        # 使用 OrderedDict 保证顺序
        self.optimizers_config = OrderedDict([
            ('MiniBatchGD', {'class': MiniBatchGD, 'params': {'lr': 7e-5}}),
            ('BatchGD', {'class': BatchGD, 'params': {'lr': 3e-4}}),
            ('AdaptiveBatchGD', {'class': AdaptiveBatchGD, 'params': {'lr': 1e-4}}),
            ('OnlineGD', {'class': OnlineGD, 'params': {'lr': 3e-4}}),
        ])
        
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

    def train_single_model(self, optimizer_name, optimizer_config):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - Optimizer: {optimizer_name}")
        
        # 重置随机种子确保公平对比
        np.random.seed(self.SEED)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        l2_scheduler = L2Scheduler(base_lambda=1e-4)
        
        # 创建优化器实例
        optimizer = optimizer_config['class'](**optimizer_config['params'])
        
        lambda_l2 = l2_scheduler.base_lambda
        
        step_losses = []  # 记录每个step的loss
        val_accuracies = []
        
        # 初始batch size
        if optimizer_name == 'OnlineGD':
            current_batch_size = 1  # Online GD每次只用一个样本
        elif optimizer_name == 'BatchGD':
            current_batch_size = train_images.shape[0]  # Batch GD使用全部数据
        else:  # MiniBatchGD 和 AdaptiveBatchGD
            current_batch_size = self.batch_size
        
        steps_per_epoch = 0  # 用于统计平均每epoch的步数
        
        for epoch in range(self.epochs):
            # AdaptiveBatchGD: 在每个epoch开始时更新batch size
            if optimizer_name == 'AdaptiveBatchGD' and epoch > 0:
                current_batch_size = optimizer.get_adaptive_batch_size()
                print(f"    Epoch {epoch}: Adaptive batch size = {current_batch_size}")
            
            # 设置dropout策略 - 从epoch 10开始启用
            if epoch < 10:
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
            epoch_steps = 0
            
            # BatchGD特殊处理：累积整个epoch的梯度
            if optimizer_name == 'BatchGD':
                optimizer.reset()  # 重置累积的梯度
                # BatchGD需要遍历所有mini-batch来累积梯度
                mini_batch_size = 64  # 使用小批量来遍历数据
                for i in range(0, shuffled_images.shape[0], mini_batch_size):
                    epoch_steps += 1
                    x = shuffled_images[i:i+mini_batch_size]
                    y = shuffled_labels[i:i+mini_batch_size]
                    
                    x = x.reshape(-1, 3, 32, 32)
                    x = augment_images(x, seed=self.SEED + epoch * 1000 + i)
                    x = x.reshape(x.shape[0], -1)
                    
                    model.zero_grad()
                    y_pred = model.forward(x, training=True)
              
                    loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
                    step_losses.append(loss)  # 记录每个step的loss
                    epoch_losses.append(loss)
                    
                    grad_output = loss_fn.backward()
                    model.backward(grad_output)
                    
                    for layer in model.layers:
                        if hasattr(layer, 'w'):
                            layer.dw += lambda_l2 * layer.w
                    
                    optimizer.accumulate_gradients(model)
                
                # BatchGD在epoch结束后统一更新
                optimizer.step(model)
            else:
                # 其他优化器正常训练
                for i in range(0, shuffled_images.shape[0], current_batch_size):
                    epoch_steps += 1
                    x = shuffled_images[i:i+current_batch_size]
                    y = shuffled_labels[i:i+current_batch_size]
                    
                    x = x.reshape(-1, 3, 32, 32)
                    x = augment_images(x, seed=self.SEED + epoch * 1000 + i)
                    x = x.reshape(x.shape[0], -1)
                    
                    model.zero_grad()
                    y_pred = model.forward(x, training=True)
              
                    loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
                    step_losses.append(loss)  # 记录每个step的loss
                    epoch_losses.append(loss)
                    
                    grad_output = loss_fn.backward()
                    model.backward(grad_output)
                    
                    for layer in model.layers:
                        if hasattr(layer, 'w'):
                            layer.dw += lambda_l2 * layer.w
                    
                    optimizer.step(model)
            
            # 统计平均步数
            steps_per_epoch += epoch_steps
            
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            val_accuracies.append(val_acc)
            
            avg_loss = np.mean(np.array(epoch_losses))
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        print(f"✅ 训练完成 - 最终测试准确率: {test_acc:.4f}")
        
        # 计算平均每epoch的步数
        avg_steps_per_epoch = steps_per_epoch // self.epochs if self.epochs > 0 else 0
        
        return {
            'step_losses': step_losses,  # 每个step的loss
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'optimizer_name': optimizer_name,
            'steps_per_epoch': avg_steps_per_epoch,
            'actual_batch_size': current_batch_size if optimizer_name != 'AdaptiveBatchGD' else 'Adaptive'
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 开始Batch Optimizer对比实验...")
        print(f"📦 基准Batch Size: {self.batch_size}")
        print(f"🔧 测试优化器及其学习率:")
        for opt_name, opt_config in self.optimizers_config.items():
            lr = opt_config['params']['lr']
            print(f"   • {opt_name}: {lr}")
        print(f"💧 Dropout启用时间: epoch >= 10")
        print("-" * 60)
        
        for i, (opt_name, opt_config) in enumerate(self.optimizers_config.items()):
            print(f"\n{'='*60}")
            print(f"实验进度: {i+1}/{len(self.optimizers_config)}")
            
            result = self.train_single_model(opt_name, opt_config)
            self.results[opt_name] = result
        
        print(f"\n🎉 所有实验完成！")
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_optimizer_comparison_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {}
        for opt_name, result in self.results.items():
            step_losses = [float(x.get() if hasattr(x, 'get') else x) for x in result['step_losses']]
            val_accuracies = [float(x.get() if hasattr(x, 'get') else x) for x in result['val_accuracies']]
            test_accuracy = float(result['test_accuracy'].get() if hasattr(result['test_accuracy'], 'get') else result['test_accuracy'])
            # 从优化器配置中获取学习率
            learning_rate = self.optimizers_config[opt_name]['params']['lr']
            results_to_save[opt_name] = {
                'step_losses': step_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_accuracy,
                'optimizer_name': result['optimizer_name'],
                'steps_per_epoch': result['steps_per_epoch'],
                'actual_batch_size': result['actual_batch_size'],
                'learning_rate': learning_rate,
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
        fig.suptitle('Batch Optimizer Comparison Results', fontsize=18, fontweight='bold', y=0.98)
        
        # 定义颜色方案 - 使用更易区分的颜色
        colors = {
            'BatchGD': '#FF6B6B',        # 红色
            'OnlineGD': '#4ECDC4',       # 青色
            'MiniBatchGD': '#FFA500',    # 橙色
            'AdaptiveBatchGD': '#9B59B6' # 紫色
        }
        
        # 1. 训练Loss曲线 (按epoch，每个epoch取平均loss)
        ax1 = axes[0, 0]
        for opt_name, result in self.results.items():
            step_losses_np = self._to_numpy(result['step_losses'])
            steps_per_epoch = result['steps_per_epoch']
            
            # 计算每个epoch的平均loss
            epoch_losses = []
            for epoch in range(self.epochs):
                start_idx = epoch * steps_per_epoch
                end_idx = min((epoch + 1) * steps_per_epoch, len(step_losses_np))
                if start_idx < len(step_losses_np):
                    epoch_loss = np.mean(step_losses_np[start_idx:end_idx])
                    epoch_losses.append(epoch_loss)
            
            epochs = range(len(epoch_losses))
            ax1.plot(epochs, epoch_losses, label=opt_name, 
                    color=colors.get(opt_name, 'gray'), linewidth=2, alpha=0.8)
        ax1.set_title('Training Loss Curves (Average per Epoch)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_xlim(left=0)  # 确保X轴从0开始
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        
        # 2. 验证Accuracy曲线
        ax2 = axes[0, 1]
        for opt_name, result in self.results.items():
            val_accuracies_np = self._to_numpy(result['val_accuracies'])
            ax2.plot(val_accuracies_np, label=opt_name, 
                    color=colors.get(opt_name, 'gray'), linewidth=2, marker='s', markersize=3)
        ax2.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # 3. 测试准确率柱状图
        ax3 = axes[1, 0]
        opt_names = list(self.results.keys())
        test_accs = [self._to_numpy(self.results[name]['test_accuracy']) for name in opt_names]
        
        bar_colors = [colors.get(name, 'gray') for name in opt_names]
        bars = ax3.bar(range(len(opt_names)), test_accs, 
                      color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Optimizer', fontsize=12)
        ax3.set_ylabel('Test Accuracy', fontsize=12)
        ax3.set_xticks(range(len(opt_names)))
        ax3.set_xticklabels(opt_names, fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='both', which='major', labelsize=10)
        
        # 在柱状图上添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, test_accs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 最终收敛性能对比表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['Optimizer', '实际Batch Size', '最终Loss', '最佳Val Acc', '测试准确率']
        
        for opt_name in opt_names:
            result = self.results[opt_name]
            actual_bs = result['actual_batch_size']
            final_loss = self._to_numpy(result['step_losses'][-1])  # 最后一个step的loss
            best_val_acc = max(self._to_numpy(result['val_accuracies']))
            test_acc = self._to_numpy(result['test_accuracy'])
            
            table_data.append([
                opt_name,
                f'{actual_bs}',
                f'{final_loss:.4f}',
                f'{best_val_acc:.4f}',
                f'{test_acc:.4f}'
            ])
        
        # 创建表格
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为suptitle留出空间
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'batch_optimizer_comparison_plots_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📈 图表已保存到: {filename}")
        
        # 尝试显示图表（如果有GUI环境）
        try:
            plt.show()
        except:
            print("📱 无GUI环境，图表已保存为文件")
        
        # 打印最佳结果
        best_opt = max(self.results.keys(), key=lambda name: self._to_numpy(self.results[name]['test_accuracy']))
        best_acc = self._to_numpy(self.results[best_opt]['test_accuracy'])
        print(f"\n🏆 最佳优化器: {best_opt} (测试准确率: {best_acc:.4f})")

def main():
    """主函数"""
    print("🎯 Batch Optimizer对比实验")
    print("=" * 60)
    
    # 创建实验对象
    experiment = BatchOptimizerExperiment()
    
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
