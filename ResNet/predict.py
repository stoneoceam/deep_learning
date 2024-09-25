import json
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt

from model import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('COCODataset/classes.json', 'r', encoding='utf-8') as file:
    classes = json.load(file)

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load image
img_path = "C:\\Users\\stone\\Desktop\\python_project\\dataset\\COCO2017\\val2017\\000000006040.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)
plt.imshow(img)

img = data_transform(img)  # [N, C, H, W]

img = torch.unsqueeze(img, dim=0)  # expand batch dimension

model = resnet50(num_classes=len(classes)).to(device)
weight_path = "params/model_16.pth"
model.load_state_dict(torch.load(weight_path, map_location=device))

model.eval()
with torch.no_grad():
    img = img.to(device)
    output = model(img)
    output = torch.squeeze(output).cpu()
    predict = torch.softmax(output, dim=0)
    predict = predict.argmax(dim=0).numpy()
    print(predict)
    pass
