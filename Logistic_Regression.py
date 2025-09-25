import numpy as np
import matplotlib.pyplot as plt

# 生成更复杂的数据，让初始化差异更明显
np.random.seed(42)
n_samples = 1000
n_features = 20  # 增加特征维度

# 生成权重和偏置
true_weights = np.random.randn(n_features) * 2
true_bias = 1.5

# 生成特征数据
X = np.random.randn(n_samples, n_features)
# 添加一些相关性，使问题更有挑战性
X[:, 1] = X[:, 0] * 0.7 + np.random.randn(n_samples) * 0.3
X[:, 3] = X[:, 2] * 0.6 - X[:, 1] * 0.4 + np.random.randn(n_samples) * 0.2

# 生成标签（加入更多噪声）
linear_model = np.dot(X, true_weights) + true_bias
y_prob = 1 / (1 + np.exp(-linear_model))
y = (y_prob + np.random.randn(n_samples) * 0.3) > 0.5
y = y.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def logistic_regression_fit(X, y, learning_rate=0.1, n_iter=1000, init_zeros=True):
    n_samples, n_features = X.shape
    if init_zeros:
        weights = np.zeros(n_features)
        print("零初始化: 所有权重从0开始")
    else:
        weights = np.random.randn(n_features) * 1.0  # 增大初始化方差
        print(f"随机初始化: 权重从 {weights[:3].round(3)}... 开始")
    
    bias = 0
    losses = []
    
    for i in range(n_iter):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        
        loss = -np.mean(y * np.log(predictions + 1e-8) + (1-y) * np.log(1-predictions + 1e-8))
        losses.append(loss)
        
        dw = (1/n_samples) * np.dot(X.T, (predictions - y))
        db = (1/n_samples) * np.sum(predictions - y)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        # 每100次迭代打印一次进度
        if i % 200 == 0:
            print(f"Iteration {i}: loss = {loss:.4f}")
    
    return weights, bias, losses

print("=== 零初始化 ===")
weights_zero, bias_zero, losses_zero = logistic_regression_fit(X, y, init_zeros=True)

print("\n=== 随机初始化 ===")
weights_rand, bias_rand, losses_rand = logistic_regression_fit(X, y, init_zeros=False)

# 绘制对比图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_zero, label='Zero Initialization', color='red', linewidth=2)
plt.plot(losses_rand, label='Random Initialization', color='blue', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Convergence Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 放大前50次迭代，观察初期差异
plt.subplot(1, 2, 2)
plt.plot(losses_zero[:50], label='Zero Initialization', color='red', linewidth=2)
plt.plot(losses_rand[:50], label='Random Initialization', color='blue', linewidth=2)
plt.xlabel('Iterations (first 50)')
plt.ylabel('Loss')
plt.title('Early Stage Convergence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('sub.png')

# 打印最终结果对比
print(f"\n最终损失 - 零初始化: {losses_zero[-1]:.6f}")
print(f"最终损失 - 随机初始化: {losses_rand[-1]:.6f}")
print(f"权重差异范数: {np.linalg.norm(weights_zero - weights_rand):.6f}")