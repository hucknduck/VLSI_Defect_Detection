import os
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from torchvision import transforms
import pandas as pd
from PIL import Image

from NewModel import CNNTiny

MODEL_PATH = "CNNTinyForDemo/Best.pth"
DATA_PATH = "DemoImages"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNTiny().to(device)
model.eval()
state_dict = torch.load(MODEL_PATH, map_location=device)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "") 
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

transform = transforms.Compose([
    transforms.ToTensor(),  
])

csv_path = os.path.join(DATA_PATH, "labels.csv")
labels_df = pd.read_csv(csv_path)

images = []
labels = []
preds = []
for i in range(4):
    img_path = os.path.join(DATA_PATH, f"image_{i}.png")
    img = Image.open(img_path).convert("L")
    tensor = transform(img).unsqueeze(0).to(device) 
    with torch.no_grad():
        logits = model(tensor)
    
    probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
    pred_label = np.argmax(probs)

    X_pred = pred_label // 48
    Y_pred = pred_label % 48

    images.append(img)
    labels.append(labels_df["label"].iloc[i])
    preds.append((X_pred, Y_pred))

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for ax, img, lbl, idx in zip(axes.ravel(), images, labels, range(4)):
    X = int(lbl[1:3])
    Y = int(lbl[4:6])

    X_pred, Y_pred = preds[idx]
    print(f"True class: {X*48 + Y}, Predicted class: {X_pred*48 + Y_pred}")
    print(f"True Coordinate: {lbl}, Predicted Coordinate: {preds[idx]}")
    ax.imshow(img, cmap="gray")
    ax.scatter(Y, X, color="red", s=100, label="True Label", marker='o')
    ax.scatter(Y_pred, X_pred, color="lime", s=100, label="Predicted Label", marker='x')
    ax.axis("off")

plt.tight_layout()
plt.show()