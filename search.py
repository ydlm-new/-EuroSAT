"""超参数搜索模块：网格搜索与随机搜索。"""
import os
import numpy as np
import itertools
from model import MLP
from trainer import train, evaluate


def grid_search(train_X, train_y, val_X, val_y, input_dim, num_classes, param_grid, epochs=30, save_dir='checkpoints'):
    """网格搜索超参数。

    param_grid示例:
    {
        'lr': [0.01, 0.05, 0.1],
        'hidden1': [256, 512],
        'hidden2': [128, 256],
        'weight_decay': [1e-4, 1e-3],
        'activation': ['relu', 'tanh'],
    }
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    results = []
    best_acc = 0.0
    best_config = None

    print(f"Grid Search: {len(combinations)} combinations to evaluate")
    print("=" * 80)

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {config}")

        model = MLP(
            input_dim=input_dim,
            hidden1=config.get('hidden1', 256),
            hidden2=config.get('hidden2', 128),
            num_classes=num_classes,
            activation=config.get('activation', 'relu'),
            seed=42
        )

        trial_save_dir = os.path.join(save_dir, f'search_trial_{i}')
        history = train(
            model, train_X, train_y, val_X, val_y,
            epochs=epochs,
            batch_size=config.get('batch_size', 128),
            lr=config.get('lr', 0.01),
            weight_decay=config.get('weight_decay', 1e-4),
            save_dir=trial_save_dir,
            verbose=False
        )

        val_acc = max(history['val_acc'])
        results.append({'config': config, 'val_acc': val_acc, 'history': history})
        print(f"  -> Best Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config

    print("\n" + "=" * 80)
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    return results, best_config


def random_search(train_X, train_y, val_X, val_y, input_dim, num_classes,
                  param_distributions, n_trials=10, epochs=30, save_dir='checkpoints', seed=42):
    """随机搜索超参数。

    param_distributions示例:
    {
        'lr': ('log_uniform', 1e-3, 0.1),
        'hidden1': ('choice', [128, 256, 512]),
        'hidden2': ('choice', [64, 128, 256]),
        'weight_decay': ('log_uniform', 1e-5, 1e-2),
        'activation': ('choice', ['relu', 'tanh']),
    }
    """
    rng = np.random.RandomState(seed)
    results = []
    best_acc = 0.0
    best_config = None

    print(f"Random Search: {n_trials} trials")
    print("=" * 80)

    for trial in range(n_trials):
        config = {}
        for key, dist in param_distributions.items():
            if dist[0] == 'choice':
                config[key] = dist[1][rng.randint(len(dist[1]))]
            elif dist[0] == 'log_uniform':
                log_low, log_high = np.log(dist[1]), np.log(dist[2])
                config[key] = float(np.exp(rng.uniform(log_low, log_high)))
            elif dist[0] == 'uniform':
                config[key] = float(rng.uniform(dist[1], dist[2]))
            elif dist[0] == 'int_uniform':
                config[key] = int(rng.randint(dist[1], dist[2] + 1))

        print(f"\n[Trial {trial+1}/{n_trials}] Config: {config}")

        model = MLP(
            input_dim=input_dim,
            hidden1=config.get('hidden1', 256),
            hidden2=config.get('hidden2', 128),
            num_classes=num_classes,
            activation=config.get('activation', 'relu'),
            seed=42
        )

        trial_save_dir = os.path.join(save_dir, f'search_trial_{trial}')
        history = train(
            model, train_X, train_y, val_X, val_y,
            epochs=epochs,
            batch_size=config.get('batch_size', 128),
            lr=config.get('lr', 0.01),
            weight_decay=config.get('weight_decay', 1e-4),
            save_dir=trial_save_dir,
            verbose=False
        )

        val_acc = max(history['val_acc'])
        results.append({'config': config, 'val_acc': val_acc, 'history': history})
        print(f"  -> Best Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config

    print("\n" + "=" * 80)
    print(f"Best Configuration: {best_config}")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    return results, best_config
