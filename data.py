from __future__ import annotations  # damit die die definition von typen erst später ausgewertet wird

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid # für Visualisierung
from math import ceil

data_path = "/home/finn/LMU/ml_ops/corruptmnist_v1"


def corrupt_mnist():
    """Return train and test datasets for corrupt MNIST."""
    train_images = []
    train_target = []
    for i in range(5):
        # alle images und targets in eine Liste laden
        train_images.append(torch.load(f"{data_path}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{data_path}/train_target_{i}.pt"))
    # aus der Liste wieder einen einzigen Tensor machen
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # testdaten laden, dafür bracuht man keine Liste
    test_images = (torch.load(f"{data_path}/test_images.pt"))
    test_target = (torch.load(f"{data_path}/test_target.pt"))

    # jetzt müssen wir noch eine dimension hinzufügen, weil CNNs die Channel dimension bracuhen, die in den Daten nicht enthalten ist
    train_images = train_images.unsqueeze(1) #fügt Dimension an zweiter Stelle ein
    train_target = train_target.long()    # ändert ins Long Zahlenformat
    test_images = test_images.unsqueeze(1) #fügt Dimension an zweiter Stelle ein
    test_target = test_target.long()      # ändert ins Long Zahlenformat

    # für training und testing jeweils einen datensatz mit images und labels haben
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set

def show_images(images: torch.tensor, target: torch.tensor) -> None:
    figure = plt.figure(figsize=(10,10))
    n = len(images)
    grid = ImageGrid(figure, 111, nrows_ncols=(ceil(n**0.5), ceil(n**0.5)), axes_pad = 0.3)
    for i, ax in enumerate(grid):
        if i <= n:
            ax.imshow(images[i].squeeze(), cmap="gray")
            ax.set_title(target[i].item())
            ax.axis("off")
    plt.show()

# aufruf 
if __name__ == "__main__":
    train_set, test_set = corrupt_mnist()
    print(f"{len(train_set)} training images of size {train_set[0][0].shape}")

    show_images(train_set.tensors[0][:25], train_set.tensors[1][:25])