# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
import boto3
import io

# 记录代码开始时间
start_time = time.time()

# 在SageMaker中访问S3
s3_client = boto3.client('s3')
bucket_name = 'uploadfromdan'
file_key = 'framingham.csv'

# 从S3读取数据
obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))

# 处理缺失值，这里采用均值填充
df = df.fillna(df.mean())

# 提取特征变量和目标变量
X = df.drop('TenYearCHD', axis=1).values
y = df['TenYearCHD'].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上进行预测
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = (y_pred > 0.5).float().numpy()
    y_test = y_test.numpy()

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'模型准确率: {accuracy:.2%}')

# 查看分类报告
print('分类报告：')
classification_report_str = classification_report(y_test, y_pred)
print(classification_report_str)
    
# 记录代码结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print(f"代码运行时间: {run_time:.4f} 秒")

# 将评估结果输出到S3
output_content = f"""CPU逻辑回归模型评估结果
=======================

运行时间: {run_time:.4f} 秒
模型准确率: {accuracy:.2%}

分类报告:
{classification_report_str}
"""

# 使用boto3将结果写入S3
output_bytes = output_content.encode('utf-8')
s3_client.put_object(
    Bucket=bucket_name,
    Key='output.txt',
    Body=output_bytes
)
    
print(f"评估结果已保存到 s3://{bucket_name}/output.txt")