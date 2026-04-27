"""训练模块：SGD优化器、学习率衰减、训练循环。"""
import numpy as np
import os


class SGD:
    """SGD优化器，支持L2正则化(weight decay已在梯度中体现)。"""

    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class LRScheduler:
    """学习率衰减策略。"""

    def __init__(self, optimizer, decay_type='step', decay_rate=0.5, decay_every=20, min_lr=1e-6):
        self.optimizer = optimizer
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_every = decay_every
        self.min_lr = min_lr
        self.initial_lr = optimizer.lr

    def step(self, epoch):
        if self.decay_type == 'step':
            if epoch > 0 and epoch % self.decay_every == 0:
                new_lr = max(self.optimizer.lr * self.decay_rate, self.min_lr)
                self.optimizer.lr = new_lr
        elif self.decay_type == 'cosine':
            new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + np.cos(np.pi * epoch / self.decay_every))
            self.optimizer.lr = max(new_lr, self.min_lr)


def evaluate(model, X, y, batch_size=512):
    """在数据集上评估模型，返回准确率和平均损失。"""
    n = len(y)
    correct = 0
    total_loss = 0.0
    num_batches = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = X[start:end]
        yb = y[start:end]
        logits = model.forward(xb)
        loss = model.loss_fn.forward(logits, yb)
        total_loss += loss * len(yb)
        preds = np.argmax(logits, axis=1)
        correct += np.sum(preds == yb)
        num_batches += 1

    accuracy = correct / n
    avg_loss = total_loss / n
    return accuracy, avg_loss


def train(model, train_X, train_y, val_X, val_y,
          epochs=50, batch_size=128, lr=0.01, weight_decay=1e-4,
          decay_type='step', decay_rate=0.5, decay_every=20,
          save_dir='checkpoints', verbose=True):
    """训练模型并保存最优权重。"""

    optimizer = SGD(model, lr=lr)
    scheduler = LRScheduler(optimizer, decay_type=decay_type,
                            decay_rate=decay_rate, decay_every=decay_every)

    os.makedirs(save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_params = None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': []
    }

    for epoch in range(epochs):
        scheduler.step(epoch)
        current_lr = optimizer.lr

        indices = np.random.permutation(len(train_y))
        epoch_loss = 0.0
        n_samples = 0

        for start in range(0, len(train_y), batch_size):
            end = min(start + batch_size, len(train_y))
            batch_idx = indices[start:end]
            xb = train_X[batch_idx]
            yb = train_y[batch_idx]

            logits = model.forward(xb)
            loss = model.compute_loss(logits, yb, weight_decay=weight_decay)
            model.backward(weight_decay=weight_decay)
            optimizer.step()

            epoch_loss += loss * len(yb)
            n_samples += len(yb)

        avg_train_loss = epoch_loss / n_samples
        train_acc, _ = evaluate(model, train_X, train_y)
        val_acc, val_loss = evaluate(model, val_X, val_y)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = model.get_params()
            model.save(os.path.join(save_dir, 'best_model.npz'))

        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if best_params is not None:
        model.set_params(best_params)

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return history
