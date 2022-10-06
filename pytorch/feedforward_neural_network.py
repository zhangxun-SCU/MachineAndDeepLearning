import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# MNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 50
batch_size = 100
learning_rate = 0.001

# 下载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# 构建神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

accuracy = 0

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向
        # 先梯度清0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch[{epoch + 1}/{num_epochs}], Step[{i+1}/{total_step}, Loss: {loss.item():.4f}]")

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 每行最大值
            _, predicted = torch.max(outputs.data, 1)
            # 计算总共有多少张图片
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if correct/total > accuracy:
            torch.save(model.state_dict(), 'model/minst.ckpt')
            accuracy = correct/total
            print(f"Accuracy of the network on the 10000 test images: {100 * correct / total}%  *")
        else:
            print(f"Accuracy of the network on the 10000 test images: {100*correct/total}%")



