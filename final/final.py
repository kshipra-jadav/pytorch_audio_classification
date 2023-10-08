import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from customdataset import UrbanSound
from cnn import CNN

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
LEARNING_RATE = 0.001
AUDIO_DIR = "/home/20bt04013/dev/datasets/UrbanSound8K/audio"
ANNOTATIONS_FILE = "/home/20bt04013/dev/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


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

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512, 
        n_mels=64
    )

    usd = UrbanSound(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, DEVICE)

    train_data_loader = DataLoader(dataset=usd, batch_size=BATCH_SIZE)

    cnn = CNN().to(device=DEVICE)

    for i in cnn.parameters():
        print(i)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data_loader, loss_fn, optimizer, DEVICE, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trained and stored")

