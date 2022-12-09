import numpy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
#
input_size = 1
output_size = 1
num_epochs = 100000
learning_rate = 0.0001
# f-v
f1 = np.array([[40063], [40085], [40105], [40123], [40141]], dtype=np.float32)
v1 = np.array([[0.52], [0.71], [0.88], [1.04], [1.19]], dtype=np.float32)

model_fv = nn.Linear(input_size, output_size).to(device)
model_fv.load_state_dict(torch.load("model/Doppler_fv.ckpt"))
criterion_fv = nn.MSELoss()
optimizer_fv = torch.optim.Adam(model_fv.parameters(), lr=learning_rate)


# # f-v
# loss_1 = np.inf
# for epoch in range(num_epochs):
#     # f-v
#     input_fv = torch.from_numpy(v1).to(device)
#     target_fv = torch.from_numpy(f1).to(device)
#     outputs_fv = model_fv(input_fv)
#     loss_fv = criterion_fv(outputs_fv, target_fv)
#     optimizer_fv.zero_grad()
#     loss_fv.backward()
#     optimizer_fv.step()
#     if(epoch + 1) % 5 == 0:
#         if loss_fv.item() < loss_1:
#             loss_1 = loss_fv.item()
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_fv.item():.4f} *")
#             torch.save(model_fv.state_dict(), 'model/Doppler_fv.ckpt')
#         else:
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_fv.item():.4f}")

#
#
# # f-v
# test_fv = np.linspace(0, 1.5, 10000, dtype=np.float32).reshape(-1, 1)
# predict_fv = model_fv(torch.from_numpy(test_fv).to(device)).detach().cpu().numpy()
# fig, ax = plt.subplots()
# plt.title("$f_n-v_n$")
# plt.plot(test_fv, predict_fv)
# plt.savefig("imgs/fv.jpg", dpi=600)
# plt.show()
# k = (predict_fv[5000][0] - predict_fv[200][0]) / (test_fv[5000][0] - test_fv[200][0])
# v = 40002 / k
# print("v=", 40002 / k)
# print((344.510 - 344.079) * 100 / 344.079)

# v-t
t = np.array([[0.05], [0.10], [0.15], [0.20], [0.25], [0.30], [0.35], [0.40]], dtype=np.float32)
vt1 = np.array([[0.63], [1.14], [1.61], [2.08], [2.59], [3.08], [3.57], [4.05]], dtype=np.float32)
vt2 = np.array([[0.64], [1.15], [1.63], [2.10], [2.57], [3.06], [3.57], [4.05]], dtype=np.float32)
vt3 = np.array([[0.64], [1.15], [1.63], [2.10], [2.57], [3.06], [3.56], [4.03]], dtype=np.float32)
vt4 = np.array([[0.63], [1.10], [1.61], [2.10], [2.57], [3.05], [3.54], [4.03]], dtype=np.float32)
model1 = nn.Linear(input_size, output_size).to(device)
model1.load_state_dict(torch.load("model/Doppler_vt1.ckpt"))
model2 = nn.Linear(input_size, output_size).to(device)
model2.load_state_dict(torch.load("model/Doppler_vt2.ckpt"))
model3 = nn.Linear(input_size, output_size).to(device)
model3.load_state_dict(torch.load("model/Doppler_vt3.ckpt"))
model4 = nn.Linear(input_size, output_size).to(device)
model4.load_state_dict(torch.load("model/Doppler_vt4.ckpt"))
criterion_1 = nn.MSELoss()
optimizer_1 = torch.optim.Adam(model1.parameters(), lr=learning_rate)
criterion_2 = nn.MSELoss()
optimizer_2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
criterion_3 = nn.MSELoss()
optimizer_3 = torch.optim.Adam(model3.parameters(), lr=learning_rate)
criterion_4 = nn.MSELoss()
optimizer_4 = torch.optim.Adam(model4.parameters(), lr=learning_rate)

loss1max = np.inf
loss2max = np.inf
loss3max = np.inf
loss4max = np.inf
# for epoch in range(num_epochs):
#     # model1
#     inputs1 = torch.from_numpy(t).to(device)
#     target1 = torch.from_numpy(vt1).to(device)
#     outputs1 = model1(inputs1)
#     loss1 = criterion_1(outputs1, target1)
#     optimizer_1.zero_grad()
#     loss1.backward()
#     optimizer_1.step()
#     if(epoch + 1) % 5 == 0:
#         if loss1.item() < loss1max:
#             loss1max = loss1.item()
#             print(f"model1: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss1.item():.4f} *")
#             torch.save(model1.state_dict(), 'model/Doppler_vt1.ckpt')
#         else:
#             print(f"model1: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss1.item():.4f}")
#
#     inputs2 = torch.from_numpy(t).to(device)
#     target2 = torch.from_numpy(vt2).to(device)
#     outputs2 = model2(inputs2)
#     loss2 = criterion_2(outputs2, target2)
#     optimizer_2.zero_grad()
#     loss2.backward()
#     optimizer_2.step()
#     if (epoch + 1) % 5 == 0:
#         if loss2.item() < loss2max:
#             loss2max = loss2.item()
#             print(f"model2: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss2.item():.4f} *")
#             torch.save(model2.state_dict(), 'model/Doppler_vt2.ckpt')
#         else:
#             print(f"model2: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss2.item():.4f}")
#
#     inputs3 = torch.from_numpy(t).to(device)
#     target3 = torch.from_numpy(vt3).to(device)
#     outputs3 = model3(inputs3)
#     loss3 = criterion_3(outputs3, target3)
#     optimizer_3.zero_grad()
#     loss3.backward()
#     optimizer_3.step()
#     if (epoch + 1) % 5 == 0:
#         if loss3.item() < loss3max:
#             loss3max = loss3.item()
#             print(f"model3: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss3.item():.4f} *")
#             torch.save(model3.state_dict(), 'model/Doppler_vt3.ckpt')
#         else:
#             print(f"model3: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss3.item():.4f}")
#
#     inputs4 = torch.from_numpy(t).to(device)
#     target4 = torch.from_numpy(vt4).to(device)
#     outputs4 = model4(inputs4)
#     loss4 = criterion_4(outputs4, target4)
#     optimizer_4.zero_grad()
#     loss4.backward()
#     optimizer_4.step()
#     if (epoch + 1) % 5 == 0:
#         if loss4.item() < loss4max:
#             loss4max = loss4.item()
#             print(f"model4: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss4.item():.4f} *")
#             torch.save(model4.state_dict(), 'model/Doppler_vt4.ckpt')
#         else:
#             print(f"model4: Epoch [{epoch + 1}/{num_epochs}], Loss: {loss4.item():.4f}")

print(model1.state_dict())
print(model2.state_dict())
print(model3.state_dict())
print(model4.state_dict())
average = (9.762 + 9.698 + 9.652 + 9.720) / 4
print((9.7922 - average) / 9.7922)
print(2 * np.pi * ((0.103 + 0.0472 / 3) / (0.475 / 0.106))**(1/2))

test = np.linspace(0, 0.5, 10000, dtype=np.float32).reshape(-1, 1)
predict1 = model1(torch.from_numpy(test).to(device)).detach().cpu().numpy()
predict2 = model2(torch.from_numpy(test).to(device)).detach().cpu().numpy()
predict3 = model3(torch.from_numpy(test).to(device)).detach().cpu().numpy()
predict4 = model4(torch.from_numpy(test).to(device)).detach().cpu().numpy()

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
axes[0, 0].plot(test, predict1)
axes[0, 1].plot(test, predict2)
axes[1, 0].plot(test, predict3)
axes[1, 1].plot(test, predict4)

plt.savefig("imgs/vt4.jpg", dpi=600)
plt.show()
