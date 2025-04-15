# 🎧 Automated Audio Captioning (AAC) using PANNs and BERT

This project implements an **Automated Audio Captioning (AAC)** model that generates natural language descriptions for audio clips. It leverages a ResNet18-based feature extractor (inspired by PANNs), a BERT tokenizer for text handling, and an LSTM decoder for sequence generation.

---

## 📌 Overview

**Goal:** Generate descriptive captions for environmental audio using a deep learning pipeline.

**Key Components:**
- **PANNs-style Audio Encoder** using pretrained ResNet18 on spectrograms
- **LSTM Decoder** for sequence modeling
- **BERT Tokenizer** to convert captions to token IDs
- **BLEU Score + Log Loss** for evaluation
- **Clotho Dataset (reduced version)** for training and testing

---

## 🧠 Model Architecture

Audio File (.wav) → Resample to 16kHz → Mel Spectrogram → Normalize and Expand Channels → ResNet18 (ImageNet pretrained) → Feature Vector (512) → Repeat for Sequence Input (Length 20) → LSTM Decoder → Linear Layer to Vocabulary Size → Token Sequence Output


---

## 💡 Requirements
```
torch>=1.12.0
torchaudio>=0.12.0
torchvision>=0.13.0
transformers>=4.26.0
pandas>=1.3.0
nltk>=3.6.0
h5py>=3.1.0
scikit-learn>=1.0.2
matplotlib>=3.4.3
librosa>=0.10.0
tqdm>=4.62.0
```


## 📁 Project Structure

project-root/ ├── dataset_reduced/ │ ├── audio/ # Folder with .wav files │ └── captions.csv # CSV file with audio-to-caption mappings ├── mel_spectrograms.hdf5 # Auto-cached Mel Spectrograms ├── main.py # Main training and evaluation script ├── aac_model.pth # Saved model checkpoint └── README.md # This documentation file


---

## 📝 Dataset Format

The dataset is a reduced version of Clotho and should be structured as:

### 📂 `dataset_reduced/`
- `audio/`: Folder containing `.wav` audio clips
- `captions.csv`: A CSV file with columns:
  - `file_name`: audio file name (e.g., `1001.wav`)
  - `caption_1` to `caption_5`: five captions per file

**Example:**
```csv
file_name,caption_1,caption_2,caption_3,caption_4,caption_5
1001.wav,"A man is speaking","Male voice speaking","Voice of a man","Man talks","Spoken words by male"
```

## 🛠️ Installation

Clone the repository
```
git clone https://github.com/your-username/audio-captioning-pytorch.git
cd audio-captioning-pytorch
```
Install dependencies
```
pip install torch torchaudio torchvision transformers pandas h5py nltk soundfile
```
NLTK Setup (for BLEU)
```
import nltk
nltk.download('punkt')
```


## 🚀 Running the Model
Train the Model 

### Option 1
```
python main.py
```
This will:

Load and preprocess audio files

Train the AAC model over 250 epochs

Store features in mel_spectrograms.hdf5

Save trained weights to aac_model.pth

### option 2
```
train(model, dataloader, criterion, log_loss_fn, optimizer, epochs=250)
torch.save(model.state_dict(), "aac_model.pth")
```


## 🧪 Evaluation 

To evaluate the model on a random audio sample and get:

✅ Predicted Caption

🟡 Ground Truth Captions

📈 BLEU Score

📉 Log Loss
```
evaluate_model(model, dataset)
```
### Example 

After training, the model evaluates one random sample and prints:

The audio file name used

The generated caption

The original 5 ground-truth captions

The BLEU Score and Log Loss

Sample Output:
```
Audio File Used for Prediction: 1021.wav
Generated Caption: a car is driving on the road
Ground Truth Captions:
- car passing by
- a vehicle driving past
- a car drives
- car noise on street
- vehicle passing on road
BLEU Score: 0.63
Final Computed Log Loss for Prediction: 1.2031
```
## 🧠 Model Architecture
```
Audio (MelSpectrogram + ResNet18)
        ↓
   Feature Vector (512-dim)
        ↓
       LSTM Decoder
        ↓
 Vocabulary-sized Output per Time Step
        ↓
     Predicted Caption
```

## 📄 Output Example
```
Audio File Used for Prediction: 123.wav
Generated Caption: a man is playing guitar

Ground Truth Captions:
- a person is strumming a guitar
- someone is playing acoustic guitar music
- guitar strings are being plucked
- a man is performing on the guitar
- a guitarist plays a melody

BLEU Score: 0.65  
Final Computed Log Loss for Prediction: 1.2345
```

## 📂 Checkpoints
Model weights after training are saved as:
```
aac_model.pth
```

### ⚙️ Technical Details

Preprocessing:
Resampling audio to 16kHz

Generating 64-band Mel spectrograms

Normalizing and interpolating to (3, 224, 224) for ResNet input

Storing in .hdf5 for speedup

Model:
Encoder: ResNet18 without classification head (pretrained)

Decoder: LSTM (hidden size 256) + Linear layer

Tokenizer: BERT tokenizer (bert-base-uncased)

Loss: CrossEntropy + Log Loss for analysis


## ✅ To Do
 Integrate HDF5 for efficient spectrogram loading

 Use BERT tokenizer for text processing

 Train LSTM decoder

 Evaluate using BLEU

 Add support for beam search decoding

 Add GUI / Streamlit app for caption generation

