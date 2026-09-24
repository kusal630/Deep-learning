# Autoencoders for Image Reconstruction, Denoising and Generation

**CS3807 – Deep Learning Laboratory | Experiment 7**  
Shiv Nadar University Chennai | B.Tech AI & DS | Semester V

---

## Overview

End-to-end study of autoencoders and their variants — **Fully Connected AE**, **Convolutional AE**, **Denoising CAE**, and **Variational Autoencoder (VAE)** — on the MNIST dataset. Focus on image reconstruction, denoising, latent space visualization, and generative modeling.

---

## Datasets

| Dataset | Purpose | Input Shape |
|---------|---------|-------------|
| MNIST (10k train, 2k test) | Digit reconstruction & generation | `(28, 28, 1)` |
| MNIST with Gaussian noise | Denoising autoencoder | `(28, 28, 1)` |
| MNIST with Salt-and-Pepper noise | Denoising autoencoder | `(28, 28, 1)` |

Pixel values normalized from `[0, 255]` to `[0, 1]`.  
Fully connected AE flattens to `784`; convolutional AE retains `28×28×1`.

---

## Project Structure

```
├── data/               # MNIST raw & processed
├── notebooks/          # AE, CAE, Denoising CAE, VAE experiments
├── src/                # models, train, evaluate, utils
├── results/            # plots, metrics, saved models
├── README.md
└── requirements.txt
```

---

## Pipeline

**Reconstruction:**
```
Image → Encoder → Latent z → Decoder → Reconstructed Image
```

**Denoising:**
```
Clean Image → Add Noise → Noisy Image → Encoder → Latent → Decoder → Denoised
```

**VAE Generation:**
```
Image → Encoder → (μ, σ) → Sample z → Decoder → Generated/Reconstructed Image
```

---

## Model Architectures

| Model | Latent Dim | Key Layers | Output Activation |
|-------|------------|------------|-------------------|
| Fully Connected AE | 16 | Dense 128→32→16→32→128→784 | Sigmoid |
| Convolutional AE | 64 (feature map) | Conv2D → MaxPool → Conv2D → Upsample → Conv2D | Sigmoid |
| Denoising CAE | Same as CAE | Same as CAE, trained on noisy input | Sigmoid |
| VAE | 2 | Conv layers → Flatten → μ, logσ² → Dense → ConvTranspose | Sigmoid |

**Training Config:** Adam (1e-3), batch 128, 20 epochs, binary cross-entropy loss.

---

## Evaluation Metrics

- **MSE** – Mean Squared Error
- **MAE** – Mean Absolute Error
- **SSIM** – Structural Similarity Index
- **Reconstruction Loss** (VAE)
- **KL Divergence** (VAE)
- **Total Loss** (VAE)
- **Parameters & Training Time**

**Required Plots:**
1. Original vs Reconstructed
2. Training/Validation Loss (AE)
3. FC-AE vs CAE reconstruction comparison
4. Clean vs Noisy vs Denoised
5. Noise level vs MSE/MAE/SSIM
6. VAE latent space (2D)
7. Randomly generated VAE images
8. Latent space interpolation
9. VAE training/validation loss
10. Reconstruction error distribution

---

## Results (to be filled)

| Model | MSE | MAE | SSIM | Parameters | Time |
|-------|-----|-----|------|------------|------|
| FC Autoencoder | — | — | — | — | — |
| Conv. Autoencoder | — | — | — | — | — |
| Denoising CAE | — | — | — | — | — |
| VAE | — | — | — | — | — |

**VAE Losses:**

| Metric | Value |
|--------|-------|
| Reconstruction Loss | — |
| KL Loss | — |
| Total Loss | — |
| Test MSE | — |
| Test MAE | — |
| Mean SSIM | — |

---

## Installation

```bash
git clone https://github.com/<your-username>/autoencoders-mnist.git
cd autoencoders-mnist
pip install -r requirements.txt
```

**Requirements:** `tensorflow>=2.10`, `numpy`, `matplotlib`, `scikit-image`, `tqdm`

---

## Usage

```bash
# Train Fully Connected AE
python src/train.py --model fc_ae --epochs 20

# Train Convolutional AE
python src/train.py --model cae --epochs 20

# Train Denoising CAE
python src/train.py --model denoising_cae --noise gaussian --sigma 0.2

# Train VAE
python src/train.py --model vae --latent_dim 2

# Evaluate
python src/evaluate.py --model cae --checkpoint results/models/cae.h5
```

---

## References

1. Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press, 2016.
2. Kingma & Welling. "Auto-Encoding Variational Bayes." *ICLR*, 2014.
3. Vincent et al. "Stacked Denoising Autoencoders." *JMLR*, 2010.
4. LeCun, Cortes, Burges. "MNIST Handwritten Digit Database."
5. [TensorFlow](https://www.tensorflow.org) | [Keras](https://keras.io) | [scikit-image](https://scikit-image.org)

---

## License

Academic coursework – Shiv Nadar University Chennai.
