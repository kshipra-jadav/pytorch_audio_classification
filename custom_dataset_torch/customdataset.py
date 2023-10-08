import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class UrbanSound(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._convert_to_mono(signal)
        signal = self._truncate(signal)
        signal = self._right_pad(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        filename = self.annotations.iloc[index, 0]

        audio_path = os.path.join(self.audio_dir, fold, filename)

        return audio_path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]
    
    def _resample(self, signal, sr):
        if sr == self.target_sample_rate:
            return signal
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
        signal = resampler(signal)
        return signal
    
    def _convert_to_mono(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            return signal
        return signal    
    
    def _truncate(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
            return signal
        return signal

    def _right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
            return signal
        return signal
    
if __name__ == "__main__":
    AUDIO_DIR = "/home/20bt04013/dev/datasets/UrbanSound8K/audio"
    ANNOTATIONS_FILE = "/home/20bt04013/dev/datasets/UrbanSound8K/metadata/UrbanSound8K.csv"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using {DEVICE} for processing...")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512, 
        n_mels=64
    )

    usd = UrbanSound(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, DEVICE)

    print(f"There are {len(usd)} samples in this dataset")

    signal, sr = usd[0]

    print("Debug!")


    