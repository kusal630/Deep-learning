# Lab 5 — CNN Training, Tuning and Transfer Learning

CS3807 Deep Learning Laboratory, Shiv Nadar University Chennai (Semester V, 2026–27).

This lab walks through training a CNN on the Oxford-IIIT Pet dataset and checks, one knob at a time, how much each design choice actually moves the needle: weight init, regularization, optimizer, hyperparameters, transfer learning and 5-fold cross-validation. Everything runs on MobileNetV2 so it finishes in reasonable time on a CPU.

## What's in here

- Notebooks / scripts for each section of the experiment (init, regularization, batch norm, optimizers, hyperparameter sweep, transfer learning, CV, final eval).
- Saved plots (`plot1.png` … `plot15.png`) used in the write-up.
- The submission PDF (built from the `.tex` in this folder).

## Dataset

Oxford-IIIT Pets — 37 cat and dog breeds, RGB images resized to 224×224×3 and normalized the way the pretrained MobileNetV2 expects.

Split I used:

| split | images |
|-------|--------|
| train | 4729   |
| val   | 1183   |
| test  | 1478   |

The test set was left completely alone until the very last evaluation.

## Headline numbers

A useful sanity check first: with 37 classes, random guessing is 1/37 ≈ 2.7%. Every from-scratch run sits right around that because those models only trained for about a minute on CPU. The real result is the fine-tuned one.

| configuration                  | val / cv acc | test acc | time (min) |
|--------------------------------|--------------|----------|------------|
| baseline (he, no reg, adam)    | 0.0270       | —        | 0.9        |
| best init: he                  | 0.0313       | —        | 0.6        |
| best reg: bn                   | 0.0287       | —        | 1.4        |
| best opt: sgd                  | 0.0389       | —        | 0.9        |
| best hyperparams (small cnn)   | 0.0304       | —        | —          |
| **fine-tuned MobileNetV2**     | **0.9110**   | **0.9127** | —        |

Final model (C4, selected by 5-fold CV):

- mean CV accuracy: 0.9110
- CV std: 0.0091
- test accuracy: 0.9127
- precision 0.9174 · recall 0.9127 · F1 0.9131
- total parameters: 2,426,725

## 5-fold cross-validation

| config | F1     | F2     | F3     | F4     | F5     | mean ± sd        |
|--------|--------|--------|--------|--------|--------|------------------|
| C1     | 0.8996 | 0.9144 | 0.8975 | 0.8953 | 0.8963 | 0.9006 ± 0.0070  |
| C2     | 0.7706 | 0.7241 | 0.7918 | 0.7854 | 0.7566 | 0.7657 ± 0.0241  |
| C3     | 0.9006 | 0.9070 | 0.8922 | 0.9133 | 0.8952 | 0.9017 ± 0.0077  |
| C4     | 0.9175 | 0.9175 | 0.9027 | 0.9197 | 0.8974 | 0.9110 ± 0.0091  |

Configs:
- C1 — feature extraction, head lr 1e-3, dropout 0.2, batch 32
- C2 — fine-tune, lr 1e-3, dropout 0.5, batch 64 (too hot, unstable)
- C3 — fine-tune, lr 1e-5, dropout 0.2, batch 32 (learns too slowly)
- C4 — fine-tune, lr 1e-4, dropout 0.2, batch 32 ← picked

## Optimizer comparison (from-scratch small CNN)

Each epoch was about 9 seconds (74 steps at ~118 ms/step), so these were short runs.

| optimizer | final loss | best val acc | epoch to converge | time (min) |
|-----------|------------|--------------|-------------------|------------|
| sgd       | 9.2183     | 0.0389       | 5                 | 0.9        |
| momentum  | 9.6351     | 0.0270       | 1                 | 0.9        |
| rmsprop   | 11.1761    | 0.0287       | 1                 | 0.9        |
| adam      | 13.3916    | 0.0270       | 1                 | 0.9        |

## Batch normalization — quick numerical check

For `x = [2, 4, 6, 8]`: mean 5.0, variance 5.0, so `x_hat ≈ [-1.342, -0.447, 0.447, 1.342]`. Matches the script output exactly.

## Running it

```bash
pip install tensorflow matplotlib scikit-learn
# then open the notebooks in order, or run the scripts section by section
```

No GPU needed; the whole thing was done on CPU.

## Build the report

```bash
pdflatex report.tex
pdflatex report.tex   # second pass for links
```

Swap each `\fbox` placeholder in the `.tex` for the matching `plotN.png` before the final compile.

## References

1. Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press, 2016.
2. Ioffe & Szegedy — Batch Normalization, ICML 2015.
3. Sandler et al. — MobileNetV2, CVPR 2018.
4. Parkhi et al. — Cats and Dogs, CVPR 2012.
5. TensorFlow docs · Keras docs
