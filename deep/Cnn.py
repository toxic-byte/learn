# ======================
# 一、导入必要的库
# ======================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ======================
# 二、MNIST 数据集概述（中文注释）
# ======================
"""
MNIST（Modified National Institute of Standards and Technology）是机器学习与计算机视觉领域的
经典入门级数据集，由 Yann LeCun 等人于 1994 年整理自 NIST 的原始手写数字数据。
它包含 60,000 张训练图像与 10,000 张测试图像，每张为 28x28 像素的灰度手写数字（0~9）。
"""

# ======================
# 三、数据预处理与加载
# ======================

# 定义数据预处理流程：
# 1. ToTensor()：将 PIL 图像或 NumPy 数组转换为 PyTorch Tensor，并将像素值从 [0, 255] 缩放到 [0.0, 1.0]
# 2. Normalize(mean, std)：对 Tensor 进行标准化，这里 mean=0.5, std=0.5，将像素从 [0,1] 映射到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 单通道灰度图，所以 mean 和 std 都是单个值
])

# 下载并加载训练集（train=True），存储到 ./data 文件夹
train_dataset = datasets.MNIST(
    root='./data',          # 数据存储路径
    train=True,             # 加载训练集
    download=True,          # 如果本地没有则自动下载
    transform=transform     # 应用上述预处理
)

# 下载并加载测试集（train=False）
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# 使用 DataLoader 进行批量加载与打乱（训练集需要 shuffle=True）
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================
# 四、定义 CNN 模型（SimpleCNN）
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：1 个输入通道（灰度图），32 个输出通道，3x3 卷积核，padding=1 保持尺寸不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：32 个输入通道，64 个输出通道，3x3 卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层：2x2 窗口，步幅为 2（图像尺寸减半）
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：64 * 7 * 7 输入（经过两次池化后图像为 7x7），输出 128 维特征
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2（输出层）：128 维输入，10 维输出（对应 0~9 十个类别）
        self.fc2 = nn.Linear(128, 10)
        # 激活函数 ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第1层：卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        # 第2层：卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        # 展平：将多维特征图展成一维向量, -1 表示自动推断 batch 维度
        x = x.view(-1, 64 * 7 * 7)  # -> [batch, 64 * 7 * 7 = 3136]
        # 全连接1 -> ReLU
        x = self.relu(self.fc1(x))  # -> [batch, 128]
        # 全连接2（输出层，不加激活，因为之后会用 CrossEntropyLoss）
        x = self.fc2(x)  # -> [batch, 10]
        return x

# ======================
# 五、初始化模型、损失函数与优化器
# ======================
model = SimpleCNN()  # 创建模型实例

# 定义损失函数：交叉熵损失（适用于多分类任务，内部包含 Softmax）
criterion = nn.CrossEntropyLoss()

# 定义优化器：Adam，学习率设置为 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ======================
# 六、模型训练过程（5个 epoch）
# ======================
print("开始训练...")
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式（影响 Dropout / BatchNorm 等层）
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播：模型预测
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 优化器更新参数
        optimizer.step()
        
        # 打印训练过程中损失（每 100 个 batch 打印一次）
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # 打印每个 epoch 的平均损失（可选）
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] 完成，平均 Loss: {avg_loss:.4f}')

print("训练完成！")

# ======================
# 七、模型评估（测试集准确率）
# ======================
model.eval()  # 设置为评估模式
correct = 0
total = 0

# 不计算梯度（eval阶段不需要反向传播）
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)           # 模型预测结果
        _, predicted = torch.max(outputs, 1)  # 取概率最大的类别索引
        total += labels.size(0)           # 累计样本总数
        correct += (predicted == labels).sum().item()  # 累计正确预测的数量

# 打印测试集准确率
accuracy = 100 * correct / total
print(f'测试集准确率: {accuracy:.2f}%')

# ======================
# 八、（可选）可视化部分预测结果
# ======================
# 可视化前10张测试图片及其预测结果
import numpy as np

# 获取一批测试数据
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images[:10], labels[:10]  # 只看前10张

# 模型预测
outputs = model(images)
_, preds = torch.max(outputs, 1)

# 反归一化，用于显示图像
images = images / 2 + 0.5  # 因为之前是 normalize 到 [-1, 1]，现在还原到 [0,1]
images = images.numpy()    # Tensor 转 numpy

# 绘制图像和预测标签
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i in range(10):
    ax = axes[i]
    ax.imshow(np.squeeze(images[i]), cmap='gray')  # 去掉多余的维度，显示灰度图
    ax.set_title(f"Pred: {preds[i].item()}")
    ax.axis('off')
plt.tight_layout()
plt.show()