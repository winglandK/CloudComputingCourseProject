# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import boto3
import io
import numpy as np
from sklearn import metrics

start_time = time.time()

# 在SageMaker中访问S3
s3_client = boto3.client('s3')
bucket_name = 'uploadfromdan'
file_key = 'Advertising Budget and Sales.csv'

# 从S3读取数据
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
data = pd.read_csv(io.BytesIO(obj['Body'].read()), header=0, index_col=0)

# 打印列名以便调试
print("CSV文件的列名:", data.columns.tolist())

# 2. 使用pandas构建X（特征向量）和y（标签）
# 修改列名以匹配实际CSV文件
feature_cols = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']
X = data[feature_cols].values
y = data['Sales ($)'].values

# 3. 构建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 移除GPU相关代码，直接使用CPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X_train = X_train.to(device)
# y_train = y_train.to(device)
# X_test = X_test.to(device)
# y_test = y_test.to(device)

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)  # 移除.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000006)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上进行预测
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.numpy()  # 移除.cpu()
    y_test = y_test.numpy()  # 移除.cpu()

# 计算均方根误差（RMSE）
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test[i])**2
sum_error = np.sqrt(sum_mean / len(y_pred))
print("均方根为:", sum_error)

# 记录代码结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print(f"代码运行时间: {run_time:.4f} 秒")

# 将评估结果输出到S3
output_content = f"""CPU线性回归模型评估结果
=======================

运行时间: {run_time:.4f} 秒
均方根误差(RMSE): {float(sum_error):.4f}  # 确保sum_error是标量

模型参数:
学习率: 0.000006
训练轮次: 1000
优化器: SGD
"""

# 使用boto3将结果写入S3
output_bytes = output_content.encode('utf-8')
s3_client.put_object(
    Bucket=bucket_name,
    Key='outputAd.txt',
    Body=output_bytes
)
    
print(f"评估结果已保存到 s3://{bucket_name}/outputAd.txt")