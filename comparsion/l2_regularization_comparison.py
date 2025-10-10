"""
L2正则化对比实验
批量训练模型，对比不同L2正则化系数的效果
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
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import Adam

class L2RegularizationExperiment:
    def __init__(self):
        self.SEED = 2023
        np.random.seed(self.SEED)
        
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        
        # 要测试的L2正则化系数
        self.l2_lambdas = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
        
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

    def train_single_model(self, lambda_l2):
        """训练单个模型"""
        print(f"\n🔧 开始训练 - L2 Lambda: {lambda_l2}")
        
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
                
                model.zero_grad()
                y_pred = model.forward(x, training=False)  # 不使用dropout
          
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=lambda_l2)
                step_losses.append(loss)  # 记录每个step的loss
                epoch_losses.append(loss)
                
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
        
        # 获取最佳验证准确率
        best_val_acc = np.max(np.array(val_accuracies))
        
        print(f"✅ 训练完成 - 最佳验证准确率: {best_val_acc:.4f}, 最终测试准确率: {test_acc:.4f}")
        
        return {
            'step_losses': step_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'lambda_l2': lambda_l2
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print(f"\n🎯 开始L2正则化对比实验...")
        print(f"📊 固定学习率: {self.learning_rate}")
        print(f"📦 固定Batch Size: {self.batch_size}")
        print(f"🔧 测试L2正则化系数: {self.l2_lambdas}")
        print(f"❌ 已关闭Dropout和数据增强")
        print("-" * 60)
        
        for i, lambda_l2 in enumerate(self.l2_lambdas):
            print(f"\n{'='*60}")
            print(f"实验进度: {i+1}/{len(self.l2_lambdas)}")
            
            result = self.train_single_model(lambda_l2)
            self.results[f'λ={lambda_l2}'] = result
        
        print(f"\n🎉 所有实验完成！")
        self.save_results()
        self.plot_results()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"l2_regularization_comparison_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_to_save = {}
        for l2_name, result in self.results.items():
            step_losses = [float(x.get() if hasattr(x, 'get') else x) for x in result['step_losses']]
            val_accuracies = [float(x.get() if hasattr(x, 'get') else x) for x in result['val_accuracies']]
            best_val_accuracy = float(result['best_val_accuracy'].get() if hasattr(result['best_val_accuracy'], 'get') else result['best_val_accuracy'])
            test_accuracy = float(result['test_accuracy'].get() if hasattr(result['test_accuracy'], 'get') else result['test_accuracy'])
            results_to_save[l2_name] = {
                'step_losses': step_losses,
                'val_accuracies': val_accuracies,
                'best_val_accuracy': best_val_accuracy,
                'test_accuracy': test_accuracy,
                'lambda_l2': result['lambda_l2'],
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"📁 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表 - 单个柱状图显示验证集和测试集准确率"""
        print(f"\n📊 生成可视化图表...")
        
        # 创建图表
        plt.close('all')  # 关闭之前的图表
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('L2 Regularization: Validation & Test Accuracy Comparison', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        # 准备数据
        l2_names = list(self.results.keys())
        val_accs = [self._to_numpy(self.results[name]['best_val_accuracy']) for name in l2_names]
        test_accs = [self._to_numpy(self.results[name]['test_accuracy']) for name in l2_names]
        
        # 设置柱状图参数
        x = np.arange(len(l2_names))  # 标签位置
        width = 0.35  # 柱子宽度
        
        # 绘制柱状图
        bars1 = ax.bar(x - width/2, val_accs, width, label='Validation Accuracy',
                      color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy',
                      color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加标签和标题
        ax.set_xlabel('L2 Regularization Lambda (λ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(l2_names, fontsize=11)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # 在柱状图上添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=9)
        
        # 设置y轴范围，留出空间显示标签
        y_max = max(max(val_accs), max(test_accs))
        ax.set_ylim(0, y_max * 1.1)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])  # 为suptitle留出空间
        
        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'l2_regularization_comparison_plots_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"📈 图表已保存到: {filename}")
        
        # 尝试显示图表（如果有GUI环境）
        try:
            plt.show()
        except:
            print("📱 无GUI环境，图表已保存为文件")
        
        # 打印最佳结果
        best_l2 = max(self.results.keys(), key=lambda name: self._to_numpy(self.results[name]['test_accuracy']))
        best_acc = self._to_numpy(self.results[best_l2]['test_accuracy'])
        print(f"\n🏆 最佳L2正则化系数: {best_l2} (测试准确率: {best_acc:.4f})")

def main():
    """主函数"""
    print("🎯 L2正则化对比实验")
    print("=" * 60)
    
    # 创建实验对象
    experiment = L2RegularizationExperiment()
    
    # 运行实验
    experiment.run_experiments()
    
    print("\n✅ 实验完成！")
    print("📊 已生成:")
    print("  - 验证集和测试集准确率对比柱状图")
    print("  - 每个L2参数显示两个柱：验证集准确率和测试集准确率")

if __name__ == "__main__":
    main()

