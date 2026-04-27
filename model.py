"""模型定义模块：三层MLP分类器，手动实现前向传播与反向传播。"""
import numpy as np


# ========== 激活函数 ==========

class ReLU:
    """ReLU激活函数。"""

    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    """Sigmoid激活函数。"""

    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


class Tanh:
    """Tanh激活函数。"""

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)


ACTIVATIONS = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}


# ========== 线性层 ==========

class Linear:
    """全连接线性层。"""

    def __init__(self, in_features, out_features, seed=None):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / in_features)
        self.W = rng.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros((1, out_features), dtype=np.float32)
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        self.grad_W = self.input.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T


# ========== Softmax + 交叉熵损失 ==========

class SoftmaxCrossEntropy:
    """Softmax + 交叉熵损失（合并计算，数值稳定）。"""

    def forward(self, logits, labels):
        self.batch_size = logits.shape[0]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        self.probs = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        self.labels = labels
        log_probs = -np.log(self.probs[np.arange(self.batch_size), labels] + 1e-12)
        return np.mean(log_probs)

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.labels] -= 1.0
        grad /= self.batch_size
        return grad


# ========== 三层MLP ==========

class MLP:
    """三层神经网络分类器。

    结构: Input -> Linear -> Activation -> Linear -> Activation -> Linear -> Softmax
    """

    def __init__(self, input_dim, hidden1, hidden2, num_classes, activation='relu', seed=42):
        self.activation_name = activation
        act_cls = ACTIVATIONS[activation]

        rng = np.random.RandomState(seed)
        seeds = rng.randint(0, 100000, size=3)

        self.fc1 = Linear(input_dim, hidden1, seed=int(seeds[0]))
        self.act1 = act_cls()
        self.fc2 = Linear(hidden1, hidden2, seed=int(seeds[1]))
        self.act2 = act_cls()
        self.fc3 = Linear(hidden2, num_classes, seed=int(seeds[2]))

        self.loss_fn = SoftmaxCrossEntropy()
        self.layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        x = self.fc3.forward(x)
        return x

    def compute_loss(self, logits, labels, weight_decay=0.0):
        ce_loss = self.loss_fn.forward(logits, labels)
        reg_loss = 0.0
        if weight_decay > 0:
            for layer in self.layers:
                reg_loss += np.sum(layer.W ** 2)
            reg_loss *= 0.5 * weight_decay
        return ce_loss + reg_loss

    def backward(self, weight_decay=0.0):
        grad = self.loss_fn.backward()
        grad = self.fc3.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.fc1.backward(grad)

        if weight_decay > 0:
            for layer in self.layers:
                layer.grad_W += weight_decay * layer.W

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x):
        logits = self.forward(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'fc{i+1}_W'] = layer.W.copy()
            params[f'fc{i+1}_b'] = layer.b.copy()
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W = params[f'fc{i+1}_W'].copy()
            layer.b = params[f'fc{i+1}_b'].copy()

    def save(self, path):
        np.savez(path, **self.get_params())

    def load(self, path):
        params = dict(np.load(path))
        self.set_params(params)
