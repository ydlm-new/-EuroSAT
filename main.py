"""主程序入口：整合数据加载、训练、超参数搜索、测试评估和可视化。"""
import numpy as np
import os
import sys
import time

from data_loader import load_dataset, preprocess, train_val_test_split, DataLoader, CLASS_NAMES
from model import MLP
from trainer import train, evaluate
from evaluate import test_model, get_misclassified, compute_confusion_matrix
from search import grid_search, random_search
from visualize import (plot_training_curves, visualize_weights,
                       visualize_confusion_matrix, visualize_errors,
                       plot_search_results)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'EuroSAT_RGB')
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
IMG_SIZE = 64
INPUT_DIM = IMG_SIZE * IMG_SIZE * 3  # 12288
NUM_CLASSES = 10


def load_and_prepare_data():
    """加载并预处理数据。"""
    print("Loading dataset...")
    images, labels = load_dataset(DATA_DIR, img_size=IMG_SIZE)
    print(f"Loaded {len(labels)} images, shape: {images.shape}")

    images_flat = preprocess(images)
    print(f"Preprocessed shape: {images_flat.shape}")

    train_X, train_y, val_X, val_y, test_X, test_y = train_val_test_split(
        images_flat, labels, val_ratio=0.15, test_ratio=0.15, seed=42)

    print(f"Train: {len(train_y)} | Val: {len(val_y)} | Test: {len(test_y)}")
    return train_X, train_y, val_X, val_y, test_X, test_y, images_flat


def run_training(train_X, train_y, val_X, val_y):
    """使用默认超参数训练模型。"""
    print("\n" + "=" * 60)
    print("TRAINING WITH DEFAULT HYPERPARAMETERS")
    print("=" * 60)

    model = MLP(
        input_dim=INPUT_DIM,
        hidden1=512,
        hidden2=256,
        num_classes=NUM_CLASSES,
        activation='relu',
        seed=42
    )

    history = train(
        model, train_X, train_y, val_X, val_y,
        epochs=50,
        batch_size=128,
        lr=0.05,
        weight_decay=1e-4,
        decay_type='step',
        decay_rate=0.5,
        decay_every=15,
        save_dir=SAVE_DIR,
        verbose=True
    )

    plot_training_curves(history, save_dir=RESULT_DIR)
    return model, history


def run_hyperparameter_search(train_X, train_y, val_X, val_y):
    """超参数搜索。"""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH")
    print("=" * 60)

    param_grid = {
        'lr': [0.01, 0.05],
        'hidden1': [256, 512],
        'hidden2': [128, 256],
        'weight_decay': [1e-4],
        'activation': ['relu', 'tanh'],
    }

    results, best_config = grid_search(
        train_X, train_y, val_X, val_y,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        param_grid=param_grid,
        epochs=20,
        save_dir=SAVE_DIR
    )

    plot_search_results(results, save_dir=RESULT_DIR)
    return results, best_config


def run_best_training(train_X, train_y, val_X, val_y, best_config):
    """使用最佳超参数重新训练。"""
    print("\n" + "=" * 60)
    print("TRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 60)
    print(f"Config: {best_config}")

    model = MLP(
        input_dim=INPUT_DIM,
        hidden1=best_config.get('hidden1', 512),
        hidden2=best_config.get('hidden2', 256),
        num_classes=NUM_CLASSES,
        activation=best_config.get('activation', 'relu'),
        seed=42
    )

    history = train(
        model, train_X, train_y, val_X, val_y,
        epochs=80,
        batch_size=128,
        lr=best_config.get('lr', 0.05),
        weight_decay=best_config.get('weight_decay', 1e-4),
        decay_type='step',
        decay_rate=0.5,
        decay_every=20,
        save_dir=SAVE_DIR,
        verbose=True
    )

    plot_training_curves(history, save_dir=RESULT_DIR)
    return model, history


def run_testing(model, test_X, test_y, load_path=None):
    """测试评估。"""
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)

    acc, cm = test_model(model, test_X, test_y, CLASS_NAMES, model_path=load_path)
    visualize_confusion_matrix(cm, CLASS_NAMES, save_dir=RESULT_DIR)
    return acc, cm


def run_analysis(model, test_X, test_y, images_flat):
    """错例分析和权重可视化。"""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    visualize_weights(model, save_dir=RESULT_DIR, img_size=IMG_SIZE)

    wrong_idx, true_labels, pred_labels = get_misclassified(model, test_X, test_y)
    print(f"\nMisclassified: {len(wrong_idx)} / {len(test_y)} ({len(wrong_idx)/len(test_y)*100:.1f}%)")

    if len(wrong_idx) > 0:
        visualize_errors(wrong_idx, true_labels, pred_labels, test_X,
                         CLASS_NAMES, save_dir=RESULT_DIR, n_show=16, img_size=IMG_SIZE)


def main():
    """主函数。"""
    start_time = time.time()

    # 1. 数据加载
    train_X, train_y, val_X, val_y, test_X, test_y, images_flat = load_and_prepare_data()

    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'

    if mode == 'train':
        model, history = run_training(train_X, train_y, val_X, val_y)
        run_testing(model, test_X, test_y)
        run_analysis(model, test_X, test_y, images_flat)

    elif mode == 'search':
        results, best_config = run_hyperparameter_search(train_X, train_y, val_X, val_y)

    elif mode == 'test':
        model = MLP(input_dim=INPUT_DIM, hidden1=512, hidden2=256,
                     num_classes=NUM_CLASSES, activation='relu')
        model_path = os.path.join(SAVE_DIR, 'best_model.npz')
        run_testing(model, test_X, test_y, load_path=model_path)
        run_analysis(model, test_X, test_y, images_flat)

    elif mode == 'full':
        # Step 1: 超参数搜索
        results, best_config = run_hyperparameter_search(train_X, train_y, val_X, val_y)

        # Step 2: 用最佳配置完整训练
        model, history = run_best_training(train_X, train_y, val_X, val_y, best_config)

        # Step 3: 测试评估（模型已包含最优权重，无需重新加载）
        run_testing(model, test_X, test_y)

        # Step 4: 错例分析与权重可视化
        run_analysis(model, test_X, test_y, images_flat)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"\nAll results saved to {RESULT_DIR}/")


if __name__ == '__main__':
    main()
