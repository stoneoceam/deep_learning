import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from tqdm import tqdm

from COCODataset.data_loading import COCODataset
from model import resnet50

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

net = resnet50()

COCO_train = COCODataset('C:\\Users\\stone\\Desktop\\python_project\\dataset\\COCO2017', mode='train')
COCO_val = COCODataset('C:\\Users\\stone\\Desktop\\python_project\\dataset\\COCO2017', mode='val')

batch_size = 32
train_loader = torch.utils.data.DataLoader(COCO_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(COCO_val, batch_size=batch_size, shuffle=True)

val_num = len(COCO_val)
classes = COCO_train.label2cat
print(classes)

# 如果需要修改分类的类别数 只需要在追后追加一个全连接层
inchannel = net.fc.in_features
net.fc = nn.Linear(in_features=inchannel, out_features=len(classes))

net.to(device)

loss_function = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略为-1的标签
optimizer = optim.Adam(net.parameters(), lr=0.001)
save_path = 'params'

epochs = 100
best_acc = 0.0
i = 0
for epoch in range(1, epochs):
    print(f'----------{epoch}/{epochs}----------')
    # train
    net.train()
    running_loss = 0.0
    step = 0
    for images, labels in tqdm(train_loader, file=sys.stdout):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1
    print(f'Epoch: {epoch:03d}, Loss: {running_loss / step:.4f}')

    # validate
    net.eval()
    acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, file=sys.stdout):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            predict = torch.max(outputs, 1)[1]
            acc += torch.eq(predict, labels).sum().item()
    acc = acc / val_num
    print(f'Epoch: {epoch:03d}, acc: {acc:.4f}')

    if acc > best_acc:
        best_acc = acc
        name = f'model_{i}.pth'
        i += 1
        torch.save(net.state_dict(), os.path.join(save_path, name))

print(f'Best acc: {best_acc:.4f}')
print('Finished Training')
