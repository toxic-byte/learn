import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    手动实现二分类交叉熵损失。
    :param y_true: 真实标签，形状 (N,)，取值 0 或 1。
    :param y_pred: 预测概率，形状 (N,)，取值 [0,1]。
    :param epsilon: 防止 log(0) 的小常数。
    :return: 平均损失。
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 避免 log(0) 或 log(1)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# 测试
y_true = np.array([0, 1, 1, 0])
y_pred = np.array([0.1, 0.9, 0.8, 0.4])
print("Binary Cross-Entropy (Manual):", binary_cross_entropy(y_true, y_pred))