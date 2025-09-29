import numpy as np

def binary_cross_entropy(y_true,y_pred):
    y_pred=np.clip(y_pred,1e-8,1-1e-8)
    loss=-np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    return loss

y_true=np.array([1,0,1,1,0])
y_pred=np.array([0.9,0.1,0.8,0.7,0.2])
loss=binary_cross_entropy(y_true,y_pred)
print(f"Binary Cross-Entropy Loss: {loss:.4f}")