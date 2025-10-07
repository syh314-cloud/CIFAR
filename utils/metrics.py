from re import L
try:
    import cupy as np
except ImportError:
    import numpy as np

def accuracy(y_pred,y_true):
    pred_labels = np.argmax(y_pred,axis=1)
    true_labels = np.argmax(y_true,axis=1)
    return np.mean(pred_labels == true_labels)

def top_k_accuracy(y_pred,y_true,k=5):
    true_labels = np.argmax(y_true,axis=1,keepdims=True)
    top_k_pred = np.argsort(y_pred,axis=1)[:,-k:]
    match = np.any(top_k_pred == true_labels,axis=1)
    return np.mean(match)

def confusion_matrix(y_pred,y_true,num_classes=10):
    pred_labels = np.argmax(y_pred,axis=1)
    true_labels = np.argmax(y_true,axis=1)
    matrix = np.zeros((num_classes,num_classes),dtype=int)
    for t,p in zip(true_labels,pred_labels):
        matrix[t,p] += 1
    return matrix

def classification_report(y_pred,y_true,class_names=None):
    cm = confusion_matrix(y_pred,y_true)
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(cm.shape[0])]
    precision = np.diag(cm) / (np.sum(cm,axis=0)+1e-8)
    recall = np.diag(cm) / (np.sum(cm,axis=1)+1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print("分类报告：")
    print("-" * 60)
    print(f"{'类别':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<8}")
    print("-" * 60)
    
    for i,class_name in enumerate(class_names):
        support = np.sum(cm,axis=1)[i]
        print(f"{class_name:<12} {precision[i]:<8.3f} {recall[i]:<8.3f} {f1[i]:<8.3f} {support:<8}")

    macro_avg_precision = np.mean(precision)
    macro_avg_recall = np.mean(recall)
    macro_avg_f1 = np.mean(f1)
    weighted_avg_precision = np.average(precision, weighted=np.sum(cm,axis=1))
    weighted_avg_recall = np.average(recall, weighted=np.sum(cm,axis=1))
    weighted_avg_f1 = np.average(f1, weighted=np.sum(cm,axis=1))
    total_support = np.sum(cm)
    overall_accuracy = np.sum(np.diag(cm)) / total_support

    print("-" * 60)
    print(f"{'宏平均':<12} {macro_avg_precision:<8.3f} {macro_avg_recall:<8.3f} {macro_avg_f1:<8.3f} {total_support:<8}")
    print(f"{'加权平均':<12} {weighted_avg_precision:<8.3f} {weighted_avg_recall:<8.3f} {weighted_avg_f1:<8.3f} {total_support:<8}")
    print("-" * 60)
    print(f"总体准确率: {overall_accuracy:.3f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'accuracy': overall_accuracy
    }

def print_confusion_matrix(cm, class_names=None):
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(cm.shape[0])]
    print("\n混淆矩阵:")
    print("真实\\预测",end="")
    for name in class_names:
        print(f"{name:>6}",end="")
    print()
    for i,name in enumerate(class_names):
        print(f"{name:<8}",end="")
        for j in range(cm.shape[1]):
            print(f"{cm[i,j]:>6}",end="")
        print()

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
