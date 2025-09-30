import numpy as np
from sklearn import metrics
def manual_auc(y_true, y_scores):
    """
    手动计算AUC，基于其概率解释。
    时间复杂度 O(m*n)，其中m是正样本数，n是负样本数。
    """
    # 将标签和分数组合在一起
    data = list(zip(y_scores, y_true))
    # 按预测分数降序排列（分数越高，越可能是正类）
    data.sort(key=lambda x: x[0], reverse=True)
    
    # 分离正样本和负样本的分数
    pos_scores = [score for score, label in data if label == 1]
    neg_scores = [score for score, label in data if label == 0]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    # 如果只有正样本或只有负样本，AUC无法定义
    if n_pos == 0 or n_neg == 0:
        return 0.5 # 或者抛出异常
    
    # 计算有多少对（正，负）中，正样本的分数 > 负样本的分数
    correct_order_pairs = 0
    for pos_score in pos_scores:
        for neg_score in neg_scores:
            if pos_score > neg_score:
                correct_order_pairs += 1
            # 如果分数相等，算作 0.5 对
            elif pos_score == neg_score:
                correct_order_pairs += 0.5
    
    # AUC = 满足条件的配对数量 / 总配对数量
    auc = correct_order_pairs / (n_pos * n_neg)
    return auc

# 测试手动实现
y_true = np.array([0, 0, 1, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
manual_auc_value = manual_auc(y_true, y_scores)
auc_value=metrics.roc_auc_score(y_true,y_scores)
print(f"AUC (manual implementation): {manual_auc_value:.4f}")
print(f"AUC ( implementation): {auc_value:.4f}")
print(f"Result matches sklearn: {np.isclose(manual_auc_value, auc_value)}")
