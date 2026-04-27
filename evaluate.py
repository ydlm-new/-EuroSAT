"""测试评估模块：准确率、混淆矩阵。"""
import numpy as np


def compute_accuracy(model, X, y, batch_size=512):
    """计算分类准确率。"""
    n = len(y)
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = model.predict(X[start:end])
        correct += np.sum(preds == y[start:end])
    return correct / n


def compute_confusion_matrix(model, X, y, num_classes=10, batch_size=512):
    """计算混淆矩阵。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n = len(y)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = model.predict(X[start:end])
        for true_label, pred_label in zip(y[start:end], preds):
            cm[true_label][pred_label] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    """格式化打印混淆矩阵。"""
    header = f"{'':>20s}" + "".join(f"{name:>12s}" for name in class_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:>20s}" + "".join(f"{cm[i][j]:>12d}" for j in range(len(class_names)))
        print(row)


def classification_report(cm, class_names):
    """打印每类的精确率、召回率、F1分数。"""
    print(f"\n{'Class':>20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 60)
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = np.sum(cm[i, :])
        print(f"{name:>20s} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

    total = np.sum(cm)
    overall_acc = np.trace(cm) / total
    print("-" * 60)
    print(f"{'Overall Accuracy':>20s} {overall_acc:>10.4f}")


def get_misclassified(model, X, y, batch_size=512):
    """获取所有分类错误的样本索引、真实标签和预测标签。"""
    all_preds = []
    n = len(y)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = model.predict(X[start:end])
        all_preds.append(preds)
    all_preds = np.concatenate(all_preds)
    wrong_mask = all_preds != y
    wrong_indices = np.where(wrong_mask)[0]
    return wrong_indices, y[wrong_indices], all_preds[wrong_indices]


def test_model(model, test_X, test_y, class_names, model_path=None):
    """完整测试流程：加载模型 -> 计算准确率 -> 打印混淆矩阵。"""
    if model_path is not None:
        model.load(model_path)
        print(f"Loaded model from {model_path}")

    acc = compute_accuracy(model, test_X, test_y)
    print(f"\nTest Accuracy: {acc:.4f}")

    cm = compute_confusion_matrix(model, test_X, test_y, num_classes=len(class_names))
    print("\nConfusion Matrix:")
    print_confusion_matrix(cm, class_names)
    classification_report(cm, class_names)

    return acc, cm
