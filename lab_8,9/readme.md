# Attention type and position in a pretrained ResNet50

Lab project for CS3807 Deep Learning Laboratory (Experiments 8 and 9), Shiv Nadar University Chennai, B.Tech AI & Data Science, semester V.

The question behind the project is simple: if you add an attention module to an ImageNet-pretrained ResNet50, does it matter which kind of attention you add, and does it matter where you put it? Everything is tested on a small five-class flower dataset, with the same split, augmentation, optimizer and training schedule for every model, so that any difference comes from the attention module and nothing else.

## What is in here

```
attention_resnet50_lab.py     the whole experiment, written to run as one Kaggle cell
README.md                     this file
outputs/                      figures, csv tables and run records (created when you run the script)
```

The script does not need any other file. Training, evaluation, timing, Grad-CAM and all the plots live in `attention_resnet50_lab.py`.

## Dataset

Five flower classes: daisy, dandelion, roses, sunflowers, tulips.

I used the Flowers Recognition dataset on Kaggle (`alxmamaev/flowers-recognition`, 4,242 images). The handout recommends the TensorFlow Flowers set (`tf_flowers`, 3,670 images), which has the same classes. If no flower dataset is attached to the notebook, the script downloads the official `tf_flowers` archive instead, which needs Internet turned on.

Images are resized to 224 x 224 and passed through the standard ResNet50 ImageNet preprocessing. The split is stratified 70 / 15 / 15 for train, validation and test with a fixed seed. It is written to `outputs/split/split.csv` and reused by every run. Augmentation (horizontal flip, small rotation, zoom, translation, mild contrast change) is the same for every model.

## Models

The backbone is always ResNet50 with ImageNet weights, followed by global average pooling, dropout (0.3) and a 5-way softmax. Attention is inserted at one of four points:

| Position | After stage | Feature map |
|----------|-------------|-------------|
| P1 | conv2_x | 56 x 56 x 256 |
| P2 | conv3_x | 28 x 28 x 512 |
| P3 | conv4_x | 14 x 14 x 1024 |
| P4 | conv5_x | 7 x 7 x 2048 |

Attention modules, all written from scratch in the script and all shape-preserving:

- Channel attention: global average pooling, a two-layer bottleneck (reduction 16) and a sigmoid gate. This is the same formula as Squeeze-and-Excitation, so it is reported as "Channel/SE".
- Spatial attention: channel-wise mean and max, concatenated, then a k x k convolution and a sigmoid (k = 7 by default).
- Channel then spatial, and spatial then channel.
- Self-attention: queries, keys and values over the flattened feature map, with a learnable residual scale that starts at zero.
- Combined: channel, spatial and self-attention in sequence.
- ECA, CBAM and Coordinate Attention, as the additional mechanisms from section 19 of the handout.

## Experiments

Seventeen runs in total, all with the same protocol.

| Group | Runs |
|-------|------|
| Baseline | A0 |
| Attention type at P3 | A1 channel, A2 spatial, A3 channel+spatial, A4 self-attention, A5 combined |
| Position, channel+spatial | B1 (P1), B2 (P2), A3 (P3), B4 (P4) |
| Multiple positions | C1 (P3+P4), C2 (P2+P3+P4) |
| Attention order | D1 spatial then channel, compared with A3 |
| Spatial kernel size | E1 (k=3), E2 (k=5), A3 (k=7) |
| Additional mechanisms at P3 | F1 ECA, F2 CBAM, F3 Coordinate Attention |

Training has two phases. For 10 epochs the backbone is frozen and only the attention module and classifier are trained at a learning rate of 1e-3. Then conv5_x is unfrozen and everything trainable is tuned for another 10 epochs at 1e-4. Batch norm statistics stay frozen throughout. The weights from the best validation epoch are kept. Loss is categorical cross-entropy and the optimizer is Adam.

## Running it

1. Open a new Kaggle notebook.
2. Settings: accelerator set to GPU T4 x2, Internet on (the ImageNet weights are downloaded).
3. Click Add Input and attach a flowers dataset, for example "Flowers Recognition".
4. Paste `attention_resnet50_lab.py` into one cell and run it.

The script uses `tf.distribute.MirroredStrategy`, so both T4s are used, along with mixed float16. The global batch size is 64. A full run of all 17 experiments should take somewhere around one to one and a half hours on 2 x T4, but I have not timed it carefully. If the session dies, run the cell again. Finished runs are detected and skipped.

Settings you may want to change are at the top of the script: `HEAD_EPOCHS`, `FT_EPOCHS`, the two learning rates, `SELECTED` (the attention model used for the confusion matrix and the main Grad-CAM comparison) and `ATT_BIAS_INIT`.

`ATT_BIAS_INIT` deserves a note. With a plain sigmoid gate, a freshly added module scales the pretrained features by about 0.5, which disturbs them before any training has happened. I initialise the gate bias to 2.0, so the module starts close to identity. Setting it to 0.0 gives the textbook initialisation.

## Outputs

Everything is written to `/kaggle/working/outputs` and zipped to `/kaggle/working/attention_lab_outputs.zip` (the large weight files are left out of the zip).

- `tables/`: the main result table, one table per experiment, per-class F1 for every run, feature dimensions before and after each attention block, Grad-CAM coverage and channel-weight statistics.
- `figures/`: sample images, accuracy and loss curves, bar charts of accuracy and macro F1, accuracy against parameters, macro F1 against inference time, confusion matrices, internal attention maps, self-attention responses, spatial attention at P1 to P4, and Grad-CAM comparisons for correct and incorrect predictions.
- `runs/`: one json record per run (history, metrics, timing) and the saved test probabilities.

For each model the script reports accuracy, macro precision, macro recall, macro F1, total and trainable parameters, parameters added over the baseline, inference time per image and training time.

## Results

Fill this in after running. Numbers below are placeholders.

| Model | Position | Accuracy | Macro F1 | Parameters | Inference (ms/img) |
|-------|----------|----------|----------|------------|--------------------|
| Baseline | none | | | | |
| Channel/SE | P3 | | | | |
| Spatial | P3 | | | | |
| Channel+Spatial | P2 | | | | |
| Channel+Spatial | P3 | | | | |
| Channel+Spatial | P4 | | | | |
| Self-attention | P3 | | | | |
| ECA | P3 | | | | |
| CBAM | P3 | | | | |
| Coordinate Attention | P3 | | | | |

## Reading the results

The test set has about 550 images, so a single image is worth roughly 0.18 accuracy points. Each configuration is trained once with one seed, and GPU training is not bit-for-bit repeatable. Differences of less than about one point between two models should not be read as one being better. The accuracy bar chart draws a 95% binomial interval for this reason.

Grad-CAM and the internal attention maps are different things. The attention maps are weights the network computes during its forward pass. Grad-CAM is built afterwards from gradients of the predicted class score. In this project Grad-CAM always uses the last feature map before global pooling, so it can be compared across models.

## Limitations

- One dataset, one backbone, one seed per configuration.
- Test-set conclusions come from a few hundred images, and the qualitative figures show five of them. The Grad-CAM coverage table over 64 images is there so that visual claims have some numbers behind them.
- Inference time is measured on a single T4 with batch size 32 and includes the host-to-device copy. It is useful for comparing models with each other, not as a deployment figure.

## References

1. He, Zhang, Ren, Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
2. Hu, Shen, Albanie, Sun, Wu. Squeeze-and-Excitation Networks. TPAMI 2020.
3. Woo, Park, Lee, Kweon. CBAM: Convolutional Block Attention Module. ECCV 2018.
4. Wang, Wu, Zhu, Li, Zuo, Hu. ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks. CVPR 2020.
5. Hou, Zhou, Feng. Coordinate Attention for Efficient Mobile Network Design. CVPR 2021.
6. Vaswani et al. Attention Is All You Need. NeurIPS 2017.
7. Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. ICCV 2017.

