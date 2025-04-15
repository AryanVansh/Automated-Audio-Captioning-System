import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from transformers import BertTokenizer, BertModel
from torchvision import models
import os
import pandas as pd
import h5py
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

# Ensure torchaudio uses soundfile backend
torchaudio.set_audio_backend("soundfile")

# Custom collate function to pad audio waveforms
def collate_fn(batch):
    audio = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    max_length = max([a.shape[1] for a in audio])
    padded_audio = [torch.nn.functional.pad(a, (0, max_length - a.shape[1]), "constant", 0) for a in audio]
    return torch.stack(padded_audio), torch.stack(captions)

# Load Pretrained Audio Feature Extractor (PANNs)
class PANNsFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.resnet18(weights="IMAGENET1K_V1")
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        return self.cnn(x)

# Text Processing with BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define AAC Model
class AACModel(nn.Module):
    def __init__(self, audio_feature_dim, hidden_dim, vocab_size):
        super().__init__()
        self.audio_encoder = PANNsFeatureExtractor()
        self.lstm = nn.LSTM(audio_feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, audio_features):
        encoded_audio = self.audio_encoder(audio_features)
        encoded_audio = encoded_audio.unsqueeze(1).repeat(1, 20, 1)
        lstm_out, _ = self.lstm(encoded_audio)
        return self.fc(lstm_out)

# HDF5 Storage Setup
hdf5_file = "mel_spectrograms.hdf5"
if not os.path.exists(hdf5_file):
    with h5py.File(hdf5_file, "w") as f:
        pass

# Dataset Class for Clotho Dataset
class ReducedClothoDataset(Dataset):
    def __init__(self, audio_dir, captions_file):
        self.audio_dir = audio_dir
        self.captions_df = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

    def __len__(self):
        return len(self.captions_df)

    def __getitem__(self, idx):
        audio_file = os.path.join(self.audio_dir, self.captions_df.iloc[idx]["file_name"].strip())
        file_name = os.path.basename(audio_file).replace(".wav", "")

        # Read from HDF5 if available
        with h5py.File(hdf5_file, "a") as f:
            if file_name in f:
                spectrogram = torch.tensor(f[file_name][:])
            else:
                try:
                    waveform, sr = torchaudio.load(audio_file)
                    waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                    spectrogram = self.mel_spectrogram(waveform)
                    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
                    spectrogram = spectrogram.expand(3, -1, -1)
                    spectrogram = torch.nn.functional.interpolate(spectrogram.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
                    
                    # Store in HDF5
                    f.create_dataset(file_name, data=spectrogram.numpy())
                except Exception as e:
                    print(f"Error loading file {audio_file}: {e}")
                    spectrogram = torch.zeros(3, 224, 224)

        captions = [self.captions_df.iloc[idx][f"caption_{i+1}"] for i in range(5)]
        caption_text = random.choice(captions)
        caption = self.tokenizer(caption_text, padding="max_length", max_length=20, truncation=True, return_tensors="pt")["input_ids"].squeeze()
        return spectrogram, caption

# Initialize Dataset and DataLoader
audio_dir = "dataset_reduced/audio/"
captions_file = "dataset_reduced/captions.csv"
dataset = ReducedClothoDataset(audio_dir, captions_file)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Initialize Model
model = AACModel(audio_feature_dim=512, hidden_dim=256, vocab_size=tokenizer.vocab_size)
criterion = nn.CrossEntropyLoss()
log_loss_fn = nn.LogSoftmax(dim=-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function with Log Loss
def train(model, dataloader, criterion, log_loss_fn, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_log_loss = 0.0
        for audio, captions in dataloader:
            optimizer.zero_grad()
            outputs = model(audio)
            outputs = outputs.view(-1, outputs.shape[-1])
            captions = captions.view(-1)
            
            loss = criterion(outputs, captions)
            log_probs = log_loss_fn(outputs)  # Apply LogSoftmax

            # Extract log probabilities of the correct classes
            log_probs_selected = log_probs[range(outputs.shape[0]), captions]

            # Compute log loss
            log_loss = -torch.sum(log_probs_selected) / outputs.shape[0]

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_log_loss += log_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_log_loss = total_log_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Log Loss: {avg_log_loss:.4f}")
    
    # Print final computed log loss
    print(f"Final Computed Log Loss: {avg_log_loss:.4f}")

train(model, dataloader, criterion, log_loss_fn, optimizer, epochs=250)
torch.save(model.state_dict(), "aac_model.pth")

# Inference & Evaluation
def generate_caption(model, audio_file):
    model.eval()
    file_name = os.path.basename(audio_file).replace(".wav", "")

    with h5py.File(hdf5_file, "r") as f:
        if file_name in f:
            spectrogram = torch.tensor(f[file_name][:])
        else:
            try:
                waveform, sr = torchaudio.load(audio_file)
                waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
                mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)(waveform)
                mel_spectrogram = (mel_spectrogram - mel_spectrogram.mean()) / mel_spectrogram.std()
                mel_spectrogram = mel_spectrogram.expand(3, -1, -1)
                spectrogram = torch.nn.functional.interpolate(mel_spectrogram.unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
            except Exception as e:
                print(f"Error processing file {audio_file}: {e}")
                return ""

    with torch.no_grad():
        output = model(spectrogram.unsqueeze(0))
    predicted_ids = torch.argmax(output, dim=-1).squeeze().tolist()
    return tokenizer.decode(predicted_ids, skip_special_tokens=True)

# Evaluate Model with BLEU Score
def evaluate_model(model, dataset):
    model.eval()
    idx = random.randint(0, len(dataset) - 1)
    audio_file = os.path.join(audio_dir, dataset.captions_df.iloc[idx]["file_name"].strip())
    file_name = os.path.basename(audio_file)  # Extract file name
    ground_truth_captions = [dataset.captions_df.iloc[idx][f"caption_{i+1}"] for i in range(5)]
    generated_caption = generate_caption(model, audio_file)

    print(f"Audio File Used for Prediction: {file_name}")
    print(f"Generated Caption: {generated_caption}")
    print("Ground Truth Captions:", ground_truth_captions)

    smoothie = SmoothingFunction().method4
    reference_tokens = [gt.split() for gt in ground_truth_captions]
    candidate_tokens = generated_caption.split()

    bleu_score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie) if candidate_tokens else 0.0
    print(f"BLEU Score: {bleu_score}")

    # Compute final log loss for the selected sample
    with torch.no_grad():
        spectrogram = dataset[idx][0].clone().detach().unsqueeze(0)  # Fix tensor warning
        output = model(spectrogram)
        log_probs = log_loss_fn(output)  

        captions = dataset[idx][1].clone().detach()  # Fix tensor warning
        captions = torch.clamp(captions, min=0, max=log_probs.shape[-1] - 1)  # Clip invalid indices
        
        # Ensure captions are within the valid range
        captions = captions[:log_probs.shape[1]]  # Truncate captions to match sequence length
        log_probs_selected = log_probs[0, range(len(captions)), captions]  # Use valid indices
        log_loss = -torch.sum(log_probs_selected) / captions.shape[0]



    print(f"Final Computed Log Loss for Prediction: {log_loss:.4f}")

evaluate_model(model, dataset)
