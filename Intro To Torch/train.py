import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
LEARNING_RATE = 0.001

def download_mnist_datasets():
    train_data = datasets.MNIST(
        download=True,
        root="mnist_data",
        train=True,
        transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        download=True,
        root="mnist_data",
        train=False,
        transform=ToTensor()
    )

    return train_data, validation_data


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)

        return predictions


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Loss :- {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch :- {i}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-----------------------")


if __name__ == "__main__":
    train_data, _ = download_mnist_datasets()

    train_data_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)

    ffnet = FeedForward().to(device=DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ffnet.parameters(), lr=LEARNING_RATE)

    train(ffnet, train_data_loader, loss_fn, optimizer, DEVICE, EPOCHS)

    torch.save(ffnet.state_dict(), "ffnet.pth")
    print("Model trained and stored")

