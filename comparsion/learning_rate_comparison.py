import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cupy as np
    import numpy as np_cpu
except ImportError:
    import numpy as np
    np_cpu = np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from datetime import datetime

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from utils.data_loader import train_images, train_labels, val_images, val_labels, test_images, test_labels
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss, L2Scheduler
from utils.classic_optimizers import Adam
from utils.data_augmentation import augment_images

class LearningRateExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        self.learning_rates = [0.00005, 0.0001, 0.0005, 0.001, 0.003, 0.01]
        
        self.results = {}
        
        self.batch_size = 64
        self.epochs = 100 
    
    def train_single_model(self, lr):
        np.random.seed(self.SEED)
        
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        l2_scheduler = L2Scheduler(base_lambda=1e-4)
        optimizer = Adam(lr, beta1=0.9, beta2=0.999)
        
        lambda_l2 = l2_scheduler.base_lambda
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.epochs):
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
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                for layer in model.layers:
                    if hasattr(layer, 'w'):
                        layer.dw += lambda_l2 * layer.w
                
                optimizer.step(model)
            
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        test_pred = model.forward(test_images, training=False)
        test_acc = np.mean(np.argmax(test_pred, axis=1) == np.argmax(test_labels, axis=1))
        
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'test_accuracy': test_acc,
            'learning_rate': lr
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 开始批量训练实验...")
        
        for i, lr in enumerate(self.learning_rates):
            print(f"\n{'='*60}")
            print(f"实验进度: {i+1}/{len(self.learning_rates)}")
            
            result = self.train_single_model(lr)
            self.results[lr] = result
        
        print(f"\n🎉 所有实验完成！")
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lr_comparison_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {}
        for lr, result in self.results.items():
            train_losses = [float(x.get() if hasattr(x, 'get') else x) for x in result['train_losses']]
            val_accuracies = [float(x.get() if hasattr(x, 'get') else x) for x in result['val_accuracies']]
            test_accuracy = float(result['test_accuracy'].get() if hasattr(result['test_accuracy'], 'get') else result['test_accuracy'])
            results_to_save[str(lr)] = {
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'test_accuracy': test_accuracy,
                'learning_rate': float(result['learning_rate'])
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"📁 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        print(f"\n📊 生成可视化图表...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('学习率对比实验结果', fontsize=16, fontweight='bold')
        
        # 1. 训练Loss曲线
        ax1 = axes[0, 0]
        for lr, result in self.results.items():
            ax1.plot(result['train_losses'], label=f'LR={lr}', linewidth=2)
        ax1.set_title('训练集 Loss 曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 验证Accuracy曲线
        ax2 = axes[0, 1]
        for lr, result in self.results.items():
            ax2.plot(result['val_accuracies'], label=f'LR={lr}', linewidth=2)
        ax2.set_title('验证集 Accuracy 曲线', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 测试准确率柱状图
        ax3 = axes[1, 0]
        lrs = list(self.results.keys())
        test_accs = [self.results[lr]['test_accuracy'] for lr in lrs]
        
        bars = ax3.bar(range(len(lrs)), test_accs, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
        ax3.set_title('测试集最终准确率对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('学习率')
        ax3.set_ylabel('测试准确率')
        ax3.set_xticks(range(len(lrs)))
        ax3.set_xticklabels([f'{lr}' for lr in lrs], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for i, (bar, acc) in enumerate(zip(bars, test_accs)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 最终收敛性能对比表格
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['学习率', '最终Loss', '最佳Val Acc', '测试准确率']
        
        for lr in lrs:
            result = self.results[lr]
            final_loss = result['train_losses'][-1]
            best_val_acc = max(result['val_accuracies'])
            test_acc = result['test_accuracy']
            
            table_data.append([
                f'{lr}',
                f'{final_loss:.4f}',
                f'{best_val_acc:.4f}',
                f'{test_acc:.4f}'
            ])
        
        # 创建表格
        table = ax4.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.2, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4ECDC4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title('性能对比总结', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'lr_comparison_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 图表已保存并显示")
        
        # 打印最佳结果
        best_lr = max(self.results.keys(), key=lambda lr: self.results[lr]['test_accuracy'])
        best_acc = self.results[best_lr]['test_accuracy']
        print(f"\n🏆 最佳学习率: {best_lr} (测试准确率: {best_acc:.4f})")

def main():
    """主函数"""
    print("🎯 学习率对比实验")
    print("=" * 60)
    
    # 创建实验对象
    experiment = LearningRateExperiment()
    
    # 运行实验
    experiment.run_experiments()
    
    print("\n✅ 实验完成！")
    print("📊 已生成:")
    print("  - 训练Loss曲线")
    print("  - 验证Accuracy曲线") 
    print("  - 测试准确率柱状图")
    print("  - 性能对比表格")

if __name__ == "__main__":
    main()
