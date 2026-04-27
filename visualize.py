"""可视化模块：训练曲线、权重可视化、错例分析。"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_training_curves(history, save_dir='results'):
    """绘制训练过程中Loss和Accuracy曲线。"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # 学习率曲线
    axes[2].plot(epochs, history['lr'], 'g-')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def visualize_weights(model, save_dir='results', img_size=64, n_channels=3):
    """将第一层隐藏层权重可视化为图像。"""
    os.makedirs(save_dir, exist_ok=True)
    W = model.fc1.W  # (input_dim, hidden1)
    n_neurons = W.shape[1]
    n_show = min(n_neurons, 64)

    cols = 8
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n_show:
            w = W[:, i].reshape(img_size, img_size, n_channels)
            w_min, w_max = w.min(), w.max()
            if w_max - w_min > 1e-8:
                w = (w - w_min) / (w_max - w_min)
            else:
                w = np.zeros_like(w)
            ax.imshow(w)
            ax.set_title(f'N{i}', fontsize=8)
        ax.axis('off')

    plt.suptitle('First Layer Weight Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight visualization saved to {save_dir}/weight_visualization.png")


def visualize_confusion_matrix(cm, class_names, save_dir='results'):
    """可视化混淆矩阵为热力图。"""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))

    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix (Normalized)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm_normalized.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_normalized[i, j] > thresh else 'black',
                    fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")


def visualize_errors(wrong_indices, true_labels, pred_labels, images_flat,
                     class_names, save_dir='results', n_show=16, img_size=64):
    """可视化分类错误的样本。"""
    os.makedirs(save_dir, exist_ok=True)
    n_show = min(n_show, len(wrong_indices))

    rng = np.random.RandomState(42)
    selected = rng.choice(len(wrong_indices), size=n_show, replace=False)

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        if i < n_show:
            idx = selected[i]
            img = images_flat[wrong_indices[idx]].reshape(img_size, img_size, 3)
            if img.max() <= 1.0:
                img = np.clip(img, 0, 1)
            else:
                img = np.clip(img / 255.0, 0, 1)
            ax.imshow(img)
            true_name = class_names[true_labels[idx]]
            pred_name = class_names[pred_labels[idx]]
            ax.set_title(f'True: {true_name}\nPred: {pred_name}', fontsize=8, color='red')
        ax.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error analysis saved to {save_dir}/error_analysis.png")


def plot_search_results(results, save_dir='results'):
    """可视化超参数搜索结果。"""
    os.makedirs(save_dir, exist_ok=True)

    accs = [r['val_acc'] for r in results]
    configs = [str(r['config']) for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(accs))
    bars = ax.bar(x, accs, color='steelblue')

    best_idx = np.argmax(accs)
    bars[best_idx].set_color('red')

    ax.set_xlabel('Configuration Index')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Hyperparameter Search Results')
    ax.set_xticks(x)
    ax.grid(axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'search_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Search results saved to {save_dir}/search_results.png")
