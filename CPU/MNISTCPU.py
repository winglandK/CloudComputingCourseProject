# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
import boto3
import io

# 记录代码开始时间
start_time = time.time()

# 在SageMaker中访问S3
s3_client = boto3.client('s3')
bucket_name = 'uploadfromdan'

# 自定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, s3_path):
        # 从S3读取数据
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        self.data = pd.read_csv(io.BytesIO(obj['Body'].read()))
        
        self.labels = torch.tensor(self.data.iloc[:, 0].values, dtype=torch.long)
        self.images = torch.tensor(self.data.iloc[:, 1:].values, dtype=torch.float32)
        self.images = self.images.view(-1, 1, 28, 28)
        # 归一化处理
        self.images = self.images / 255.0  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# 加载S3数据集
train_dataset = MNISTDataset('s3://uploadfromdan/mnist_train.csv')
test_dataset = MNISTDataset('s3://uploadfromdan/mnist_test.csv')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.conv2_drop(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# 初始化模型 - 移除GPU相关代码
model = CNN()

# 定义损失函数和优化器
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 训练模型
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 移除.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 测试模型
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # 移除.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy

# 训练和测试循环
accuracies = []
for epoch in range(1, 5):
    train(model, train_loader, optimizer, epoch)
    accuracy = test(model, test_loader)
    accuracies.append(accuracy)
    
# 记录代码结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print(f"代码运行时间: {run_time:.4f} 秒")

# 将评估结果输出到S3
output_content = f"""CPU CNN模型评估结果 (MNIST)
=======================

运行时间: {run_time:.4f} 秒
最终准确率: {accuracies[-1]:.2f}%

每轮准确率:
{', '.join([f'第{i+1}轮: {acc:.2f}%' for i, acc in enumerate(accuracies)])}

模型参数:
- 学习率: 0.01
- 动量: 0.5
- 批次大小: 64
- 训练轮次: 4
"""

# 使用boto3将结果写入S3
output_bytes = output_content.encode('utf-8')
s3_client.put_object(
    Bucket=bucket_name,
    Key='ouputCPU_MNIST.txt',
    Body=output_bytes
)
    
print(f"评估结果已保存到 s3://{bucket_name}/ouputCPU_MNIST.txt")