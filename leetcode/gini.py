import numpy as np

def find_best_feature(data):
    """找到使基尼指数最小的特征索引"""
    data = np.array(data)
    n_features = data.shape[1] - 1
    
    def gini_index(feature_idx):
        """计算某个特征的基尼指数"""
        feature_col = data[:, feature_idx]
        labels = data[:, -1]
        
        gini = 0
        for value in np.unique(feature_col):
            mask = feature_col == value
            subset_labels = labels[mask]
            weight = mask.sum() / len(data)
            
            # 计算子集的基尼值: 1 - Σ(pi^2)
            _, counts = np.unique(subset_labels, return_counts=True)
            probs = counts / counts.sum()
            gini += weight * (1 - np.sum(probs ** 2))
        
        return gini
    
    # 计算所有特征的基尼指数并返回最小的
    gini_indices = [gini_index(i) for i in range(n_features)]
    return np.argmin(gini_indices)


if __name__ == "__main__":
    # 读取输入并解析
    data = eval(input())
    
    # 计算并输出结果
    result = find_best_feature(data)
    print(result)