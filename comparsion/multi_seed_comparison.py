"""
多随机种子对比实验
固定配置：
- L2正则化：λ=5e-4
- 数据增强：Crop + Flip
- 其他超参数保持一致

测试5个不同的随机种子，评估模型稳定性
每个模型使用完整的metrics评估
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
from utils.metrics import accuracy, top_k_accuracy, confusion_matrix, classification_report, print_confusion_matrix, CIFAR10_CLASSES
from model.mlp_4layer import MLP
from utils.loss import CrossEntropyLoss
from utils.classic_optimizers import Adam

class MultiSeedExperiment:
    def __init__(self):
        # 固定超参数
        self.learning_rate = 7e-05
        self.batch_size = 64
        self.epochs = 100
        
        # 固定的正则化配置
        self.l2_lambda = 5e-4
        self.use_augmentation = True
        
        # 测试的随机种子
        self.seeds = [2023, 2024, 2025, 2026, 2027]
        
        self.results = {}
    
    def _to_numpy(self, data):
        """将CuPy数组转换为NumPy数组"""
        if hasattr(data, 'get'):
            return data.get()
        elif isinstance(data, list):
            return [self._to_numpy(item) for item in data]
        else:
            return data
    
    def _apply_augmentation(self, x, seed):
        """应用Crop + Flip数据增强"""
        x = x.reshape(-1, 3, 32, 32)
        np.random.seed(seed)
        x = random_flip(x.copy(), prob=0.5)
        x = random_crop(x.copy(), crop_size=32, padding=4)
        return x.reshape(x.shape[0], -1)
    
    def train_single_model(self, seed):
        """训练单个模型"""
        print(f"\n{'='*80}")
        print(f"🔧 开始训练 - 随机种子: {seed}")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 创建模型
        model = MLP(train_images.shape[1], 1024, 512, 256, train_labels.shape[1])
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(self.learning_rate, beta1=0.9, beta2=0.999)
        
        # 关闭Dropout
        model.dropout1.p = 0.0
        model.dropout2.p = 0.0
        model.dropout3.p = 0.0
        
        step_losses = []
        val_accuracies = []
        
        for epoch in range(self.epochs):
            np.random.seed(seed + epoch)
            idx = np.random.permutation(train_images.shape[0])
            shuffled_images = train_images[idx]
            shuffled_labels = train_labels[idx]
            
            epoch_losses = []
            
            for i in range(0, shuffled_images.shape[0], self.batch_size):
                x = shuffled_images[i:i+self.batch_size]
                y = shuffled_labels[i:i+self.batch_size]
                
                # 应用数据增强
                if self.use_augmentation:
                    x = self._apply_augmentation(x, seed + epoch * 1000 + i)
                
                model.zero_grad()
                y_pred = model.forward(x, training=False)
                
                loss = loss_fn.forward(y_pred, y, model, lambda_l2=self.l2_lambda)
                step_losses.append(loss)
                epoch_losses.append(loss)
                
                grad_output = loss_fn.backward()
                model.backward(grad_output)
                
                # 添加L2正则化梯度
                if self.l2_lambda > 0:
                    for layer in model.layers:
                        if hasattr(layer, 'w'):
                            layer.dw += self.l2_lambda * layer.w
                
                optimizer.step(model)
            
            # 验证准确率
            val_pred = model.forward(val_images, training=False)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(val_labels, axis=1))
            
            # 转换为Python float
            if hasattr(val_acc, 'get'):
                val_acc_scalar = float(val_acc.get())
            else:
                val_acc_scalar = float(val_acc)
            
            val_accuracies.append(val_acc_scalar)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = float(np.mean(np.array(epoch_losses)))
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc_scalar:.4f}")
        
        # ========== 使用 metrics 进行完整评估 ==========
        print(f"\n{'='*60}")
        print("开始详细评估...")
        print(f"{'='*60}")
        
        # 1. 训练集评估
        print("\n📊 训练集评估:")
        train_pred = model.forward(train_images, training=False)
        train_acc = accuracy(train_pred, train_labels)
        train_top5 = top_k_accuracy(train_pred, train_labels, k=5)
        
        if hasattr(train_acc, 'get'):
            train_acc_scalar = float(train_acc.get())
            train_top5_scalar = float(train_top5.get())
        else:
            train_acc_scalar = float(train_acc)
            train_top5_scalar = float(train_top5)
        
        print(f"  准确率: {train_acc_scalar:.4f}")
        print(f"  Top-5准确率: {train_top5_scalar:.4f}")
        
        # 2. 验证集评估
        print("\n📊 验证集评估:")
        val_pred = model.forward(val_images, training=False)
        val_acc = accuracy(val_pred, val_labels)
        val_top5 = top_k_accuracy(val_pred, val_labels, k=5)
        
        if hasattr(val_acc, 'get'):
            val_acc_scalar = float(val_acc.get())
            val_top5_scalar = float(val_top5.get())
        else:
            val_acc_scalar = float(val_acc)
            val_top5_scalar = float(val_top5)
        
        print(f"  准确率: {val_acc_scalar:.4f}")
        print(f"  Top-5准确率: {val_top5_scalar:.4f}")
        
        # 3. 测试集评估
        print("\n📊 测试集评估:")
        test_pred = model.forward(test_images, training=False)
        test_acc = accuracy(test_pred, test_labels)
        test_top5 = top_k_accuracy(test_pred, test_labels, k=5)
        
        if hasattr(test_acc, 'get'):
            test_acc_scalar = float(test_acc.get())
            test_top5_scalar = float(test_top5.get())
        else:
            test_acc_scalar = float(test_acc)
            test_top5_scalar = float(test_top5)
        
        print(f"  准确率: {test_acc_scalar:.4f}")
        print(f"  Top-5准确率: {test_top5_scalar:.4f}")
        
        # 4. 混淆矩阵和分类报告
        print("\n📋 测试集详细分类报告:")
        test_pred_np = self._to_numpy(test_pred)
        test_labels_np = self._to_numpy(test_labels)
        
        report = classification_report(test_pred_np, test_labels_np, class_names=CIFAR10_CLASSES)
        
        cm = confusion_matrix(test_pred_np, test_labels_np)
        print_confusion_matrix(self._to_numpy(cm), class_names=CIFAR10_CLASSES)
        
        print(f"\n✅ 训练完成 - 种子: {seed}")
        print(f"   训练准确率: {train_acc_scalar:.4f}")
        print(f"   验证准确率: {val_acc_scalar:.4f}")
        print(f"   测试准确率: {test_acc_scalar:.4f}")
        
        # 转换metrics中的NumPy数组
        precision = [float(x) for x in self._to_numpy(report['precision'])]
        recall = [float(x) for x in self._to_numpy(report['recall'])]
        f1 = [float(x) for x in self._to_numpy(report['f1'])]
        cm_list = [[int(x) for x in row] for row in self._to_numpy(report['confusion_matrix'])]
        
        return {
            'seed': seed,
            'step_losses': [float(x) for x in self._to_numpy(step_losses)],
            'val_accuracies': val_accuracies,
            'train_accuracy': train_acc_scalar,
            'train_top5_accuracy': train_top5_scalar,
            'val_accuracy': val_acc_scalar,
            'val_top5_accuracy': val_top5_scalar,
            'test_accuracy': test_acc_scalar,
            'test_top5_accuracy': test_top5_scalar,
            'best_val_accuracy': max(val_accuracies),
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'confusion_matrix': cm_list,
            'steps_per_epoch': len(train_images) // self.batch_size
        }
    
    def run_experiments(self):
        """运行所有实验"""
        print("="*80)
        print("多随机种子对比实验")
        print("="*80)
        print(f"固定配置:")
        print(f"  - Learning Rate: {self.learning_rate}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - L2 Lambda: {self.l2_lambda}")
        print(f"  - Data Augmentation: Crop + Flip")
        print(f"\n测试随机种子: {self.seeds}")
        print("="*80)
        
        # 训练每个种子
        for seed in self.seeds:
            result = self.train_single_model(seed)
            self.results[f"seed_{seed}"] = result
        
        self.save_results()
        self.plot_results()
        self._print_summary()
    
    def save_results(self):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_seed_results_{timestamp}.json"
        
        save_data = {
            'experiment_config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'l2_lambda': self.l2_lambda,
                'use_augmentation': self.use_augmentation,
                'seeds': self.seeds
            },
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 结果已保存到: {filename}")
    
    def plot_results(self):
        """绘制对比图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fig = plt.figure(figsize=(20, 12))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # 1. 验证准确率曲线
        ax1 = plt.subplot(3, 3, 1)
        for idx, (seed_name, result) in enumerate(self.results.items()):
            ax1.plot(range(1, self.epochs + 1),
                    result['val_accuracies'],
                    label=f"Seed {result['seed']}",
                    color=colors[idx % len(colors)],
                    linewidth=2, alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Validation Accuracy', fontsize=11)
        ax1.set_title('Validation Accuracy Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. 测试集准确率对比
        ax2 = plt.subplot(3, 3, 2)
        
        seeds_list = [result['seed'] for result in self.results.values()]
        test_accs = [result['test_accuracy'] for result in self.results.values()]
        
        x = self._to_numpy(np.arange(len(seeds_list)))
        bars = ax2.bar(x, test_accs, color=colors[:len(seeds_list)], 
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax2.set_ylabel('Test Accuracy', fontsize=11)
        ax2.set_title('Test Accuracy by Seed', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Seed\n{s}' for s in seeds_list], fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_ylim([0, 1.0])
        
        # 3. Top-5准确率对比
        ax3 = plt.subplot(3, 3, 3)
        
        test_top5 = [result['test_top5_accuracy'] for result in self.results.values()]
        
        bars = ax3.bar(x, test_top5, color=colors[:len(seeds_list)],
                      alpha=0.8, edgecolor='black', linewidth=0.8)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax3.set_ylabel('Top-5 Accuracy', fontsize=11)
        ax3.set_title('Test Top-5 Accuracy by Seed', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Seed\n{s}' for s in seeds_list], fontsize=9)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.set_ylim([0, 1.0])
        
        # 4. 训练集、验证集、测试集准确率对比
        ax4 = plt.subplot(3, 3, 4)
        
        train_accs = [result['train_accuracy'] for result in self.results.values()]
        val_accs = [result['val_accuracy'] for result in self.results.values()]
        
        width = 0.25
        x_pos = self._to_numpy(np.arange(len(seeds_list)))
        
        bars1 = ax4.bar(x_pos - width, train_accs, width, label='Train',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars2 = ax4.bar(x_pos, val_accs, width, label='Val',
                       color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=0.8)
        bars3 = ax4.bar(x_pos + width, test_accs, width, label='Test',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.8)
        
        ax4.set_ylabel('Accuracy', fontsize=11)
        ax4.set_title('Train/Val/Test Accuracy Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Seed\n{s}' for s in seeds_list], fontsize=9)
        ax4.legend(loc='best', framealpha=0.9)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_ylim([0, 1.0])
        
        # 5. 准确率稳定性（均值和标准差）
        ax5 = plt.subplot(3, 3, 5)
        
        mean_test = np.mean(test_accs)
        std_test = np.std(test_accs)
        
        ax5.bar(['Test Accuracy'], [mean_test], color='#2ecc71', alpha=0.8, 
               edgecolor='black', linewidth=0.8)
        ax5.errorbar(['Test Accuracy'], [mean_test], yerr=[std_test],
                    fmt='none', color='black', capsize=10, linewidth=2)
        
        ax5.text(0, mean_test + std_test + 0.02,
                f'Mean: {mean_test:.4f}\nStd: {std_test:.4f}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax5.set_ylabel('Accuracy', fontsize=11)
        ax5.set_title('Test Accuracy Stability', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax5.set_ylim([0, 1.0])
        
        # 6. 每类平均Precision
        ax6 = plt.subplot(3, 3, 6)
        
        # 计算每个类别的平均precision
        avg_precision = []
        for class_idx in range(10):
            class_precisions = [result['precision_per_class'][class_idx] for result in self.results.values()]
            avg_precision.append(np.mean(class_precisions))
        
        x_classes = self._to_numpy(np.arange(10))
        bars = ax6.bar(x_classes, avg_precision, color='#3498db', alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        
        ax6.set_xlabel('Class', fontsize=11)
        ax6.set_ylabel('Average Precision', fontsize=11)
        ax6.set_title('Average Precision per Class', fontsize=12, fontweight='bold')
        ax6.set_xticks(x_classes)
        ax6.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
        ax6.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax6.set_ylim([0, 1.0])
        
        # 7. 每类平均Recall
        ax7 = plt.subplot(3, 3, 7)
        
        avg_recall = []
        for class_idx in range(10):
            class_recalls = [result['recall_per_class'][class_idx] for result in self.results.values()]
            avg_recall.append(np.mean(class_recalls))
        
        bars = ax7.bar(x_classes, avg_recall, color='#e74c3c', alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        
        ax7.set_xlabel('Class', fontsize=11)
        ax7.set_ylabel('Average Recall', fontsize=11)
        ax7.set_title('Average Recall per Class', fontsize=12, fontweight='bold')
        ax7.set_xticks(x_classes)
        ax7.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
        ax7.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax7.set_ylim([0, 1.0])
        
        # 8. 每类平均F1
        ax8 = plt.subplot(3, 3, 8)
        
        avg_f1 = []
        for class_idx in range(10):
            class_f1s = [result['f1_per_class'][class_idx] for result in self.results.values()]
            avg_f1.append(np.mean(class_f1s))
        
        bars = ax8.bar(x_classes, avg_f1, color='#2ecc71', alpha=0.8,
                      edgecolor='black', linewidth=0.8)
        
        ax8.set_xlabel('Class', fontsize=11)
        ax8.set_ylabel('Average F1 Score', fontsize=11)
        ax8.set_title('Average F1 Score per Class', fontsize=12, fontweight='bold')
        ax8.set_xticks(x_classes)
        ax8.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
        ax8.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax8.set_ylim([0, 1.0])
        
        # 9. 准确率分布箱线图
        ax9 = plt.subplot(3, 3, 9)
        
        data_for_box = [test_accs]
        bp = ax9.boxplot(data_for_box, labels=['Test Accuracy'],
                        patch_artist=True, showmeans=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.6)
        
        ax9.set_ylabel('Accuracy', fontsize=11)
        ax9.set_title('Test Accuracy Distribution', fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax9.set_ylim([0, 1.0])
        
        plt.tight_layout()
        
        plot_filename = f"multi_seed_plots_{timestamp}.png"
        plt.savefig(plot_filename, bbox_inches='tight')
        print(f"📊 图表已保存到: {plot_filename}")
        plt.close()
    
    def _print_summary(self):
        """打印实验总结"""
        print(f"\n{'='*80}")
        print("实验总结")
        print(f"{'='*80}")
        
        # 收集所有测试准确率
        test_accs = [result['test_accuracy'] for result in self.results.values()]
        test_top5 = [result['test_top5_accuracy'] for result in self.results.values()]
        
        mean_test = np.mean(test_accs)
        std_test = np.std(test_accs)
        min_test = np.min(test_accs)
        max_test = np.max(test_accs)
        
        mean_top5 = np.mean(test_top5)
        std_top5 = np.std(test_top5)
        
        print("\n📊 测试集准确率统计:")
        print(f"  均值: {mean_test:.4f}")
        print(f"  标准差: {std_test:.4f}")
        print(f"  最小值: {min_test:.4f}")
        print(f"  最大值: {max_test:.4f}")
        print(f"  范围: {max_test - min_test:.4f}")
        
        print("\n📊 测试集Top-5准确率统计:")
        print(f"  均值: {mean_top5:.4f}")
        print(f"  标准差: {std_top5:.4f}")
        
        print("\n📋 各随机种子详细结果:")
        print("-"*80)
        for seed_name, result in sorted(self.results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            print(f"  种子 {result['seed']:4d}: Test Acc={result['test_accuracy']:.4f}, "
                  f"Top-5={result['test_top5_accuracy']:.4f}, Best Val={result['best_val_accuracy']:.4f}")
        
        # 找出最佳和最差种子
        best_seed_name = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        worst_seed_name = min(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        
        print(f"\n🏆 最佳种子: {best_seed_name[1]['seed']} (测试准确率: {best_seed_name[1]['test_accuracy']:.4f})")
        print(f"⚠️ 最差种子: {worst_seed_name[1]['seed']} (测试准确率: {worst_seed_name[1]['test_accuracy']:.4f})")
        
        # 稳定性评估
        cv = (std_test / mean_test) * 100  # 变异系数
        print(f"\n📈 模型稳定性评估:")
        print(f"  变异系数 (CV): {cv:.2f}%")
        if cv < 1.0:
            print(f"  ✅ 模型非常稳定")
        elif cv < 2.0:
            print(f"  ✅ 模型稳定性良好")
        else:
            print(f"  ⚠️ 模型稳定性一般，建议进一步调优")
        
        print(f"{'='*80}\n")

def main():
    experiment = MultiSeedExperiment()
    experiment.run_experiments()

if __name__ == "__main__":
    main()

