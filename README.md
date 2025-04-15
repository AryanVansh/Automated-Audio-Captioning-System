# 🎵 AAC - Audio Captioning with BERT and PANNs

This repository implements an **Audio Captioning** system using a combination of:
- **PANNs (Pretrained Audio Neural Networks)** for extracting audio features,
- **BERT** tokenizer for text processing,
- and an **LSTM-based decoder** for generating textual captions for audio clips.

It is trained and evaluated on a reduced version of the [Clotho Dataset](https://zenodo.org/record/4783391).

---

## 📁 Dataset Structure

Expected folder structure:

dataset_reduced/ ├── audio/ │ ├── audio_file_1.wav │ ├── audio_file_2.wav │ └── ... └── captions.csv


- `captions.csv` should contain columns: `file_name`, `caption_1`, ..., `caption_5`.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/audio-captioning-aac.git
cd audio-captioning-aac
