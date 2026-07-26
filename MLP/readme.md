# Experiment 2: MLP for Fashion-MNIST Classification

Multi-layer perceptron built with Keras to classify Fashion-MNIST images into 10 clothing categories, with a baseline model and a hyperparameter search on top of it.

## Files

- `experiment2_mlp_fashion_mnist.py` — main script, runs on CPU/GPU. Loads the data, trains a baseline MLP, then runs `RandomizedSearchCV` (via scikeras) to tune hidden layers, neurons, learning rate, optimizer, and dropout.
- `experiment2_mlp_tpu_search.py` — same idea but set up for a TPU runtime (Colab). Detects the TPU, builds the model inside `strategy.scope()`, and does the train/val split manually since `validation_split` doesn't work well with TPU.

## Requirements

```
tensorflow
scikeras
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:
```
pip install tensorflow scikeras scikit-learn pandas numpy matplotlib seaborn
```

## Running it

```
python experiment2_mlp_fashion_mnist.py
```

Fashion-MNIST downloads automatically the first time via `tf.keras.datasets.fashion_mnist`. No dataset file to fetch manually.

For the TPU version, run it in a Colab notebook with the TPU runtime selected — it falls back to CPU/GPU automatically if no TPU is found.

## What it does

1. Loads Fashion-MNIST, shows a few sample images and the class distribution (6,000 images per class, so it's balanced).
2. Flattens the 28x28 images to 784-length vectors and normalizes pixel values to [0,1].
3. Trains a baseline MLP (2 hidden layers, 128 and 64 neurons) for 10 epochs.
4. Runs a randomized hyperparameter search over layer count, neuron count, learning rate, optimizer, activation, and dropout.
5. Compares baseline vs. tuned model accuracy, and plots training curves + a confusion matrix for the tuned model.

## Notes on runtime

The search space in the main script is kept small on purpose (6 random combos × 3-fold CV = 18 fits) so it finishes in a few minutes instead of taking an hour. `verbose=2` is set on `RandomizedSearchCV` so you can see each fit progress instead of it looking frozen. If you want a more thorough search, widen `param_dist` and bump up `n_iter`/`cv`, just expect it to take longer.

## Results (roughly)

- Baseline test accuracy: ~88-89%
- Tuned model test accuracy: ~91%
- Best configs consistently use `relu` + `adam`; `sigmoid` and high learning rates with `sgd` tend to underperform.
- Most confusion happens between visually similar classes — Pullover, Coat, and Shirt get mixed up with each other most often.
