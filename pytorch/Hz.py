import numpy as np
import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #正常显示负号

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size = 1
hidden_size = 500
output_size = 1
num_epochs = 300000
learning_rate = 0.001

x = np.linspace(1, 82, 82, dtype=np.float32).reshape(82, 1)
# print(x)

# y = np.array([[0.000], [0.000], [0.000], [0.000], [0.000], [0.000], [0.000], [0.000], [0.004], [0.088],
#               [0.249], [0.520], [0.761], [0.738], [0.959], [1.070], [1.134], [1.138], [1.036], [0.897],
#               [0.678], [0.419], [0.251], [1.001]
#               [3.102], [1.372], [3.954]], dtype=np.float32)

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.082, 0.19, 0.32, 0.447, 0.561, 0.639, 0.670, 0.646,
                           0.578, 0.467, 0.336, 0.223, 0.181, 0.322, 0.711, 1.124, 1.423, 1.562, 1.512, 1.282, 0.921,
                           0.531, 0.227, 0.093, 0.254, 0.787, 1.474, 2.030, 2.366, 2.464, 2.315, 1.955, 1.396, 0.799,
                           0.339, 0.158, 0.496, 1.282, 2.100, 2.722, 3.124, 3.260, 3.146, 2.759, 2.169, 1.488, 0.878,
                           0.588, 0.938, 1.685, 2.470, 3.126, 3.638, 3.886, 3.948, 3.755, 3.321, 2.720, 2.056, 1.588,
                           1.564, 2.001, 2.670, 3.334, 3.922, 4.399, 4.686, 4.734, 4.627, 4.288, 3.784], dtype=np.float32).reshape(82, 1)
# print(y)
#
# model = make_interp_spline(x.reshape(-1), y.reshape(-1), k=2)
#
# xs = np.linspace(0, 85, 1000)
# ys = model(xs)
#
# fig, ax =plt.subplots()
# ax.plot(xs, ys)
# ax.scatter(x, y, c='r')
# ax.set_xlim(0, 85)
# ax.set_ylim(0, 4)
# plt.show()

class HzModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HzModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size*2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 10)

        self.fc4 = nn.Linear(hidden_size*10, hidden_size * 5)
        self.fc5 = nn.Linear(hidden_size*5, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out

model = HzModel(input_size, hidden_size).to(device)
model.load_state_dict(torch.load("model/Hz.ckpt"))
model.eval()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# loss_1 = 1.5
# for epoch in range(num_epochs):
#     inputs = torch.from_numpy(x).to(device)
#     targets = torch.from_numpy(y).to(device)
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 5 == 0:
#         if loss.item() < loss_1:
#             loss_1 = loss.item()
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} *")
#             torch.save(model.state_dict(), 'model/Hz.ckpt')
#         else:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
#
#     if epoch % 10000 == 0:
#
#         test = np.linspace(0, 82, 10000, dtype=np.float32).reshape([-1, 1])
#         predicted = model(torch.from_numpy(test).to(device)).detach().cpu().numpy()
#         fig, ax = plt.subplots()
#         ax.scatter(x, y, c='r')
#         ax.set_ylim(0, 5)
#         ax.plot(test, predicted)
#         plt.show()

test = np.linspace(0, 82, 120, dtype=np.float32).reshape([-1, 1])
predicted = model(torch.from_numpy(test).to(device)).detach().cpu().numpy()

model_scipy = make_interp_spline(test.reshape(-1), predicted.reshape(-1), k=2)

xs = np.linspace(0, 82, 10000)
ys = model_scipy(xs)

fig, ax = plt.subplots()

# 设置坐标轴
ax.set_title("$I_A - U_{G_2K}$(手动测试)")
x_major_locator = plt.MultipleLocator(5)
y_major_locator = plt.MultipleLocator(0.5)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.set_xlabel("$U_{G_2K} / V$")
ax.set_ylabel("$I_A / 10^{-7}A$")

ax.set_ylim(0, 5, 0.1)
ax.plot(xs, ys, lw=0.8)
ax.scatter(x, y, c='r', s=1)

# 显示文字

for i in range(10, len(x), 6):
    # plt.text(x[i] - 5, y[i] + 0.3, f"({x.reshape(-1)[i]}, {y.reshape(-1)[i] :.2f})")
    ax.annotate(f"({x.reshape(-1)[i]},{y.reshape(-1)[i] :.2f})", xy=(x[i], y[i]), xytext=(x[i] - 5, y[i] + 0.3), arrowprops=dict(facecolor='blue', connectionstyle="arc3", width=1, headwidth=1))
ax.grid(lw=0.15)
plt.savefig('imgs/Hz.jpg', dpi=600)
plt.show()
