CS3807 Lab 4: Transfer Learning & CNN Architectures
This repo covers Experiment 4 for the Deep Learning Lab. The main goal here was to move beyond basic CNNs and look at Transfer Learning using ResNet50, then compare it against classic architectures like LeNet, AlexNet, VGG16, and InceptionV3 on the CIFAR-10 dataset.
We ran this on Kaggle using dual T4 GPUs, so there's some MirroredStrategy setup in the first cell to make sure both cards actually get used (otherwise Keras just sits on GPU 0).
What this notebook does
The workflow follows the lab manual tasks pretty closely:

    Data Prep: Since the Kaggle dataset is in the raw pickled format (not the easy .npz), I wrote a small helper to unpack the batches. I also resized the images from 32x32 up to 96x96 because ResNet struggles with the tiny originals.
    Transfer Learning (ResNet50):
        Phase 1 (Head Training): Loaded ImageNet weights, froze the base, and trained just the dense head for 10 epochs.
        Phase 2 (Fine Tuning): Unfroze the last block (conv5) and fine-tuned with a tiny learning rate (1e-5) so we didn't wreck the pretrained features.
    Architecture Comparison: Built and trained LeNet-5, AlexNet, VGG16, and InceptionV3 (as a stand-in for GoogleNet) to see how they stack up against the fine-tuned ResNet.
    Evaluation: Generated confusion matrices, classification reports, and a comparison table to analyze accuracy vs. parameter count.

How to run it

    Upload the CIFAR-10 Python dataset
     to your Kaggle notebook.
    Go to Settings -> Accelerator and select GPU T4 x2.
    Check the DATA_DIR variable in the second cell. It should match the path in the "Input" panel on the right.
    Run all cells.

Key Results

    ResNet50 (Fine-tuned): ~89.8% Test Accuracy.
    Best Scratch Model: AlexNet/VGG16 usually come in around 80-85% depending on the run.
    Observation: Transfer learning converges way faster. The head-only training gets us to ~85% in just a few minutes, and fine-tuning squeezes out that last 4-5%.

Files

    experiment_4_notebook.ipynb: The main code, plots, and analysis.
    README.md: This file.

References

    He, K., et al. "Deep Residual Learning for Image Recognition."
    Krizhevsky, A., et al. "ImageNet Classification with Deep Convolutional Neural Networks."
    TensorFlow/Keras Documentation.
