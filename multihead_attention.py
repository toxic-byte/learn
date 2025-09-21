import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def multi_head_self_attention(X, num_heads, W_q, W_k, W_v, W_o):
    """
    X: (batch_size, seq_len, d_model) np.ndarray
    num_heads: int
    W_q, W_k, W_v, W_o: (d_model, d_model) np.ndarray
    返回: (batch_size, seq_len, d_model) np.ndarray
    """
    batch_size, seq_len, d_model = X.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_head = d_model // num_heads

    # 1. 线性变换得到 Q, K, V
    Q = X @ W_q  # (B, L, D)
    K = X @ W_k
    V = X @ W_v

    # 2. 分头
    Q = Q.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, H, L, Dh)
    K = K.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, H, L, Dh)
    V = V.reshape(batch_size, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)  # (B, H, L, Dh)

    # 3. 注意力分数与输出
    # scores: (B, H, L, L)
    scores = (Q @ np.transpose(K, (0, 1, 3, 2))) / (d_head ** 0.5)
    attention = softmax(scores, axis=-1)
    out = attention @ V  # (B, H, L, Dh)

    # 4. 拼接并输出线性层
    out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)  # (B, L, D)
    out = out @ W_o  # (B, L, D)
    return out

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(2, 4, 8).astype(np.float32)
    num_heads = 2
    d_model = 8

    W_q = np.eye(d_model, dtype=np.float32)
    W_k = np.eye(d_model, dtype=np.float32)
    W_v = np.eye(d_model, dtype=np.float32)
    W_o = np.eye(d_model, dtype=np.float32)

    out = multi_head_self_attention(X, num_heads, W_q, W_k, W_v, W_o)
    print(out.shape)  # (2, 4, 8)
    print(out)