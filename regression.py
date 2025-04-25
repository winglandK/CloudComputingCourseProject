import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

start_time = time.time()

# 1. 读入数据
data = pd.read_csv("/root/Advertising Budget and Sales.csv", header=0, index_col=0)

# 2. 使用pandas构建X（特征向量）和y（标签）
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols].values
y = data['Sales'].values

# 3. 构建训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim).to(device)

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
    y_pred = y_pred.cpu().numpy()
    y_test = y_test.cpu().numpy()

# 计算均方根误差（RMSE）
from sklearn import metrics
import numpy as np
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test[i])**2
sum_error = np.sqrt(sum_mean / len(y_pred))
print("均方根为:", sum_error)


#绘制预测和测试集曲线
#def show_roc():
#    plt.figure()
#    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
#    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
#    plt.legend(loc="upper right")
#    plt.xlabel("The number of sales")
#    plt.ylabel("Value of sales")
#    plt.show()

#show_roc() 


# 记录代码结束时间
end_time = time.time()

# 计算运行时间
run_time = end_time - start_time
print(f"代码运行时间: {run_time:.4f} 秒")