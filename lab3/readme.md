# CS3807 Experiment 3: CNNs for Image Classification

Implementation of a Convolutional Neural Network for CIFAR-10 classification, done for the Deep Learning Laboratory course (CS3807). The notebook walks through the convolution operation, pooling, feature map visualization, and builds a small CNN from scratch using TensorFlow/Keras.

## What it covers

The script follows the lab manual tasks one by one:

- Loading CIFAR-10 from the raw pickle batches and inspecting the data
- Comparing kernel sizes (3x3, 5x5, 7x7) and their effect on feature map size
- Studying stride and padding (same vs valid) and computing output dimensions
- Visualizing the first-layer feature maps
- Building the CNN (Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> Dense -> Softmax)
- Comparing max pooling against average pooling
- Training with Adam for 20 epochs, batch size 32
- Evaluation with accuracy, precision, recall, F1, confusion matrix and classification report

All eight mandatory plots are generated inline, each with a short inference written below it for the report.

## Running it

The code is written for a Kaggle notebook with GPU acceleration.

1. Add the dataset [cifar10-python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) to your notebook (or upload the `cifar-10-batches-py` folder).
2. In the notebook settings, set Accelerator to **GPU T4 x2**.
3. Update `data_dir` at the top of the script if your mount path differs:
   ```python
   data_dir = '/kaggle/input/datasets/pankrzysiu/cifar10-python/cifar-10-batches-py'
   ```
4. Run all cells.

Both GPUs are used through `tf.distribute.MirroredStrategy`, with the model built inside the strategy scope. The global batch of 32 is split across the two devices automatically.

## Requirements

- tensorflow >= 2.10
- numpy
- matplotlib
- seaborn
- scikit-learn

All of these come preinstalled on Kaggle, so no extra setup is needed there. For a local run, `pip install tensorflow numpy matplotlib seaborn scikit-learn` is enough.

## Project layout

```
.
├── experiment3_cnn.ipynb   # the full notebook / script
└── README.md
```

The dataset itself is not committed; it is loaded from the Kaggle input path at runtime.

## Results summary

With the architecture in the manual (two conv blocks, no dropout), the network reaches roughly 65-70% validation accuracy after 20 epochs. Max pooling consistently beats average pooling by a few points. The confusion matrix shows most errors happen between visually similar classes (cat/dog, car/truck, airplane/bird), which is expected at 32x32 resolution without data augmentation.

A note on the two GPUs: for a network this small, MirroredStrategy gives almost no wall-clock speedup because gradient syncing between devices costs about as much as the compute it saves. It only starts paying off for larger models or bigger batches.

## Additional exercises

Answers to the extra questions from the manual, also present as comments in the code:

1. Output size for 64x64 input, 5x5 kernel, stride 2, padding 2: `(64 - 5 + 4)/2 + 1 = 32.5`. Not an integer, so this exact configuration is not realizable; TensorFlow would floor it to 32 or the padding needs adjusting.
2. Trainable parameters for 64 filters of 3x3 on an RGB input: `(3*3*3 + 1) * 64 = 1792`.
3. ReLU vs Sigmoid: ReLU does not saturate for positive inputs, so gradients keep flowing in deep networks. Sigmoid squashes everything into (0, 1) and its gradient vanishes near both ends, which slows or stops learning in deep stacks.
4. Average vs Max pooling: same output sizes, but average pooling dilutes strong feature responses and gives lower accuracy.
5. Increasing filters from 16 to 64 roughly quadruples the first-layer parameters and slows each epoch; accuracy improves slightly but overfitting gets worse without regularization.

## Discussion

Short answers to the six discussion questions:

1. Convolution reuses the same weights across the whole image. A fully connected layer would need 32x32x3 weights per neuron, which explodes for images.
2. Stride s shrinks the feature map by roughly a factor of s in each direction.
3. Padding controls the output size and stops border pixels from being dropped after repeated convolutions.
4. Pooling downsamples the maps, cutting parameters and adding some translation tolerance.
5. Each map is the response of one filter, so bright regions mark where that feature (edge, corner, texture) appears in the input.
6. Weight sharing means the parameter count depends on the kernel size, not the image size, which is why CNNs need far fewer parameters than MLPs.

## References

- Goodfellow et al., Deep Learning
- Bishop, Pattern Recognition and Machine Learning
- Haykin, Neural Networks and Learning Machines
- TensorFlow Documentation
- CIFAR-10 Dataset Documentation
