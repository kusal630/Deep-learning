# RNN, LSTM & GRU for Sequence Learning and Video Understanding

**CS3807 – Deep Learning Laboratory | Experiment 6**  
Shiv Nadar University Chennai | B.Tech AI & DS | Semester V

---

## Overview

End-to-end study of recurrent sequence models (Vanilla RNN, LSTM, GRU) on the UCI HAR dataset, extended to video understanding using CNN-extracted features, plus a synthetic seq2seq reversal task.

---

## Datasets

| Dataset | Purpose | Input Shape |
|---------|---------|-------------|
| UCI HAR (raw inertial signals) | Activity classification | `(N, 128, 9)` |
| UCF101 subset (3–5 classes) | Video understanding | `(10, D)` per video |
| Synthetic integer sequences | Seq2seq reversal | Variable length |

---

## Project Structure

```
├── data/               # raw & processed data
├── notebooks/          # preprocessing, visualization, training, seq2seq
├── src/                # data_loader, models, train, evaluate, utils
├── results/            # plots, metrics, saved models
├── README.md
└── requirements.txt
```

---

## Pipeline

**HAR Classification:**
```
Raw Signals → Window (128×9) → Normalize → RNN/LSTM/GRU → Dense → Softmax
```

**Video Understanding:**
```
Video → Frames → MobileNetV2 (frozen) → Features (10×D) → LSTM/GRU → Dense → Softmax
```

**Seq2Seq:**
```
Input Seq → Encoder → Context → Decoder → Output Seq
```

---

## Model Architectures

| Model | Recurrent Layer | Units | Output |
|-------|----------------|-------|--------|
| RNN   | SimpleRNN      | 32    | 6      |
| LSTM  | LSTM           | 32    | 6      |
| GRU   | GRU            | 32    | 6      |

**Training Config:** Adam (1e-3), batch 32, 30 epochs, dropout 0.2, sparse categorical cross-entropy.

---

## Evaluation Metrics

Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix, Parameters, Training Time.

**Required Plots:** Loss/Accuracy curves, Confusion matrices, Model comparison bar plot, Sequence length vs F1, Video sample frames, Video training curves, Video confusion matrix.

---

## Results (to be filled)

| Metric | RNN | LSTM | GRU |
|--------|-----|------|-----|
| Accuracy (%) | — | — | — |
| Macro F1 (%) | — | — | — |
| Parameters | — | — | — |
| Training Time (s) | — | — | — |

---

## Installation

```bash
git clone https://github.com/<your-username>/rnn-lstm-gru-sequence-learning.git
cd rnn-lstm-gru-sequence-learning
pip install -r requirements.txt
```

**Requirements:** `tensorflow>=2.10`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `opencv-python`, `tqdm`

---

## Usage

```bash
# Preprocess HAR
python src/data_loader.py --dataset har --window 128

# Train
python src/train.py --model lstm --epochs 30 --batch_size 32

# Evaluate
python src/evaluate.py --model lstm --checkpoint results/models/lstm.h5

# Video task
python src/train.py --task video --model gru --frames 10

# Seq2seq
jupyter notebook notebooks/05_seq2seq.ipynb
```

---

## References

1. Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press, 2016.
2. Hochreiter & Schmidhuber. "Long Short-Term Memory." *Neural Computation*, 1997.
3. Cho et al. "Learning Phrase Representations using RNN Encoder-Decoder." *EMNLP*, 2014.
4. Anguita et al. "A Public Domain Dataset for HAR Using Smartphones." *ESANN*, 2013.
5. Soomro, Zamir, Shah. "UCF101." 2012.

---

## License

Academic coursework – Shiv Nadar University Chennai.
