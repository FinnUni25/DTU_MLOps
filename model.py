from torch import nn
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Bilder visualisieren
#matplotlib.use("TkAgg")
#images = torch.load("/home/finn/LMU/ml_ops/corruptmnist_v1/train_images_0.pt")
#plt.imshow(images[0].squeeze(), cmap="gray")
#plt.axis("off")
#plt.show()


class MyAwesomeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # reduziert bild dimensionalität um 2 da Boxgröße 3
        self.conv2 = nn.Conv2d(32, 64, 4) # reduziert Bildgröße um 3
        self.conv3 = nn.Conv2d(64, 64, 2) # reduziert Bildgröße um 1
        self.linear1 = nn.Linear(64*2*2, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) # dim [batch_size, 32, 26, 26]
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2) # dim [batch_size, 32, 13, 13]
        x = self.conv2(x) # dim [batch_size, 64, 10, 10]
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2) # dim [batch_size, 64, 5, 5]
        x = self.conv3(x) # dim [batch_size, 64, 4, 4]
        x = torch.relu(x)
        x = torch.max_pool2d(x,2) # dim [batch_size, 64, 2, 2]
        x = x.view(x.size(0), -1) # dim [batch_size, 64*2*2]
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x
    
if __name__ == "__main__":   # folgender code wird nur ausgeführt wenn man dieses Programm ausführt, nicht wenn man meinModell importiert
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
