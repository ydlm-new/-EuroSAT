"""数据加载与预处理模块"""
import os
import numpy as np
from PIL import Image


CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def load_dataset(data_dir, img_size=64):
    """加载EuroSAT数据集，返回图像数组和标签数组。"""
    images = []
    labels = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert('RGB')
            if img.size != (img_size, img_size):
                img = img.resize((img_size, img_size))
            images.append(np.array(img, dtype=np.float32))
            labels.append(CLASS_TO_IDX[class_name])
    images = np.array(images)
    labels = np.array(labels, dtype=np.int64)
    return images, labels


def preprocess(images):
    """将图像归一化到[0,1]并展平为向量。"""
    images = images / 255.0
    n = images.shape[0]
    return images.reshape(n, -1)


def train_val_test_split(images, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按比例划分训练集、验证集和测试集，保持类别比例。"""
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx],
            images[test_idx], labels[test_idx])


class DataLoader:
    """小批量数据迭代器。"""

    def __init__(self, X, y, batch_size=128, shuffle=True, seed=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        n = len(self.y)
        indices = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (len(self.y) + self.batch_size - 1) // self.batch_size
