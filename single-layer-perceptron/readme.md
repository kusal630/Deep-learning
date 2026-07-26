# Experiment 1: Single Layer Perceptron (Banknote Authentication)

Perceptron built from scratch in numpy, used to classify banknotes as authentic or forged based on features extracted from wavelet-transformed images. Compared against scikit-learn's `Perceptron` at the end.

## File

- `experiment1_perceptron.py` — loads the dataset, does some EDA, implements the perceptron manually, trains and evaluates it, and runs a few comparison experiments (learning rates, 2D decision boundary, scikit-learn vs custom).

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Running it

```
python experiment1_perceptron.py
```

The dataset is pulled directly from the UCI repository at runtime (`data_banknote_authentication.txt`), so there's no local file to download first — just needs an internet connection.

## What it does

1. Loads the banknote dataset (variance, skewness, curtosis, entropy, class) and checks shape/missing values/basic stats.
2. EDA: histograms, a correlation heatmap, a variance-vs-skewness scatter plot, and boxplots.
3. Scales features with `StandardScaler` and splits into train/test (80/20).
4. Implements `PerceptronScratch` — a perceptron with a step activation, trained with the classic perceptron update rule (`weights += lr * (y_true - y_pred) * x`).
5. Trains it, evaluates accuracy/precision/recall/F1/confusion matrix, and plots training error, weight evolution, and bias evolution over epochs.
6. Repeats training at three learning rates (0.001, 0.01, 0.1) to compare convergence speed.
7. Trains a 2-feature version (variance, skewness) just to plot an actual 2D decision boundary.
8. Trains scikit-learn's `Perceptron` on the same data for a sanity check against the from-scratch version.

## Results (roughly)

- Both the custom perceptron and scikit-learn's version land around 98-99% accuracy.
- Weights differ slightly between the two since scikit-learn adds a bit of default regularization, but the classification outcome is basically the same.
- Learning rate 0.01 is the best trade-off — 0.1 converges fastest but can overshoot, 0.001 is just slow.
- Variance vs skewness alone gives a cleanly separable 2D plot, which is why the perceptron converges to zero training error within the first ~10 epochs.
