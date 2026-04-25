import matplotlib.pyplot as plt
import torch
import typer
from torch import nn, optim
from data import corrupt_mnist
from model import MyAwesomeModel

app = typer.Typer()

######## run code with Terminal: uv run main.py train

@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 8) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"specs: lr = {lr}, batch size =  {batch_size}, epochs = {epochs}")

    model = MyAwesomeModel()
    train_set, _ = corrupt_mnist()

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)

    calc_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    stats = {"train_loss": [], "train_acc": []}
    model.train() # setzt modell in den Trainingsmodus
    for i in range(epochs):
        for j, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            log_probs = model(images)
            loss = calc_loss(log_probs, labels)
            loss.backward()
            optimizer.step()

            stats["train_loss"].append(loss.item())
            _, predicted = torch.max(log_probs, 1)
            acc = (predicted == labels).float().mean()
            stats["train_acc"].append(acc.item())

            if (j%100==0):
                print(f"epoch {i} of {epochs}, loss: {loss.item()}, acc: {acc.item()}")

    torch.save(model.state_dict(), "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(stats["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(stats["train_acc"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_stats.png")


@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    _, test_set = corrupt_mnist()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)
    
    model.eval() # setzt modell in den Evaluationsmodus
    correct = 0
    total = 0
    with torch.no_grad(): #deaktiviert gradienten, da nicht gebraucht
        for images, labels in test_loader:
            log_probs = model(images)
            _, predicted = torch.max(log_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test accuracy: {correct/total}")


if __name__ == "__main__":
    app()
