# Face Generation with Generative Adversarial Networks (GAN)

A beginner-friendly PyTorch project that trains a **vanilla Generative Adversarial Network (GAN)** to generate synthetic face images inspired by the CelebA dataset.

The project contains a Jupyter notebook that loads and preprocesses face images, defines a generator and discriminator, trains the two networks adversarially, and displays generated samples after each epoch.

> **Note:** This notebook implements a fully connected (dense-layer) GAN, not a DCGAN. The accompanying class slides introduce both architectures, but this implementation uses linear layers.

## Table of Contents

- [Project Overview](#project-overview)
- [How a GAN Works](#how-a-gan-works)
- [Model Architecture](#model-architecture)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Requirements](#requirements)
- [Setup and Run](#setup-and-run)
- [Training Configuration](#training-configuration)
- [Output](#output)
- [Limitations and Possible Improvements](#limitations-and-possible-improvements)
- [Project Structure](#project-structure)
- [Learning Goals](#learning-goals)
- [Dataset Acknowledgment](#dataset-acknowledgment)

## Project Overview

Generative Adversarial Networks use two neural networks trained in competition:

- **Generator (G):** receives a random latent vector and tries to produce a realistic-looking face.
- **Discriminator (D):** receives real and generated images and predicts whether an image is real.

As training progresses, the generator attempts to fool the discriminator, while the discriminator learns to identify generated images.

### Workflow

1. Load face images from the dataset directory.
2. Crop, resize, convert and normalize the images.
3. Create batches using PyTorch `DataLoader`.
4. Generate fake images from random noise.
5. Train the discriminator on real and fake images.
6. Train the generator to make fake images more likely to be classified as real.
7. Display a grid of generated images after each epoch.

## How a GAN Works

```text
Random latent vector z
        |
        v
  Generator (G)
        |
        v
  Generated face ------------------+
                                   |
Real face from dataset -------------+--> Discriminator (D)
                                          |
                                          v
                                  Realness score (0 to 1)
```

The discriminator is trained with these targets:

- Real images → `1`
- Generated images → `0`

When training the generator, generated images are assigned the target `1`. This encourages the generator to produce samples that the discriminator believes are real.

## Model Architecture

### Generator

The generator starts with a 100-dimensional random vector and maps it through fully connected layers:

| Layer | Output features | Activation |
|---|---:|---|
| Input latent vector | 100 | — |
| Linear | 256 | ReLU |
| Linear | 512 | ReLU |
| Linear | 1,024 | ReLU |
| Linear | 12,288 | Tanh |
| Reshape | `3 × 64 × 64` | — |

The final output contains `64 × 64 × 3 = 12,288` values per image. `Tanh` keeps output values in the range `[-1, 1]`, matching the normalization applied to training images.

### Discriminator

The discriminator flattens each RGB image into a vector and predicts a single score:

| Layer | Output features | Activation |
|---|---:|---|
| Input image | `3 × 64 × 64` | — |
| Flatten | 12,288 | — |
| Linear | 1,024 | LeakyReLU (0.2) |
| Linear | 512 | LeakyReLU (0.2) |
| Linear | 256 | LeakyReLU (0.2) |
| Linear | 1 | Sigmoid |

The sigmoid output is between `0` and `1` and is used as the discriminator's estimate that an image is real.

### Loss and Optimization

- **Loss:** Binary Cross Entropy (`torch.nn.BCELoss`)
- **Optimizer:** Adam for both networks
- **Learning rate:** `0.0002`
- **Adam betas:** `(0.5, 0.999)`
- **Latent vector size:** `100`
- **Image size:** `64 × 64`
- **Image channels:** `3` (RGB)

The discriminator is updated to distinguish real images from fake images. The generator is updated to make the discriminator classify generated images as real.

## Dataset and Preprocessing

The notebook expects a CelebA-style image directory at:

```text
./img_align_celeba/
```

The notebook does **not** download the dataset automatically. Obtain CelebA through its official project resources and follow the dataset's terms and conditions. Extract the image files so that the `img_align_celeba` directory is beside the notebook.

Each image is processed with the following transformations:

1. `CenterCrop(178)` — crop the center to a square.
2. `Resize(64)` — resize to `64 × 64`.
3. `ToTensor()` — convert the image into a PyTorch tensor.
4. `Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))` — scale RGB values to approximately `[-1, 1]`.

The notebook reports 202,599 images when the expected full set is present. The custom dataset class attempts to open each entry in the directory as an image, so keep unrelated files and subdirectories out of that folder.

## Requirements

- Python 3
- PyTorch
- Torchvision
- Pillow
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the dependencies in your environment:

```bash
python -m pip install torch torchvision pillow numpy matplotlib jupyter
```

The notebook checks for a supported Apple MPS device first, then CUDA, and otherwise falls back to CPU. CPU training may be considerably slower.

## Setup and Run

1. **Clone your repository:**

   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```bash
   python -m pip install torch torchvision pillow numpy matplotlib jupyter
   ```

4. **Place the dataset** in `./img_align_celeba/`, next to the notebook.

5. **Launch Jupyter:**

   ```bash
   jupyter notebook
   ```

6. Open `11 DOWNLOAD GAN CODE.ipynb` and run the cells in order, from top to bottom.

The notebook's final training cell calls the training function for **5 epochs**. It prints generator and discriminator losses periodically and displays eight generated samples in a `4 × 2` grid at the end of each epoch.

## Training Configuration

The notebook uses these key settings:

```python
batch_size = 128
latent_dim = 100
image_size = 64
epochs = 5
learning_rate = 0.0002
```

The dataset loader shuffles images each epoch. The final batch can contain fewer than 128 images, and the training loop uses the actual batch size for labels and noise.

Training time depends on the device, dataset location, and system resources. The full CelebA dataset and dense fully connected layers can require substantial memory and compute. If resources are limited, test first with a smaller subset and a smaller batch size.

## Output

After each epoch, the notebook displays a grid of eight generated RGB images. Early outputs are typically noisy or unrealistic; the model needs training to learn useful visual patterns.

The notebook currently **displays** these grids but does not save them to disk. To save a figure, add the following line before `plt.show()` inside `save_generated_images`:

```python
plt.savefig(f"generated_epoch_{epoch + 1}.png", bbox_inches="tight")
```

Loss values are useful for monitoring training, but they do not by themselves prove that generated faces are realistic or diverse. Inspect sample grids over time when evaluating progress.

## Limitations and Possible Improvements

This is an educational baseline and has several limitations:

- **Fully connected image model:** flattening images ignores much of their spatial structure and uses many parameters. A convolutional GAN or DCGAN is a natural next step.
- **Short training run:** five epochs are intended as an initial experiment, not a guarantee of high-quality image generation.
- **No checkpoints:** the notebook does not currently save model weights or optimizer states for resuming training.
- **No fixed evaluation noise:** a fixed latent batch would make it easier to compare generated images across epochs.
- **Basic evaluation:** the notebook logs losses and displays samples but does not calculate formal image-quality metrics.
- **Dataset file assumptions:** the dataset loader assumes every entry in the folder is a valid image file.

Possible next steps include saving checkpoints, using fixed latent vectors for evaluation, filtering image files, adding reproducibility seeds, and implementing a DCGAN with convolutional and upsampling layers.

## Project Structure

A suggested repository layout:

```text
.
├── 11 DOWNLOAD GAN CODE.ipynb
├── README.md
└── img_align_celeba/       # Dataset; do not commit the full dataset
```

Consider renaming the notebook to something clearer, such as `gan_face_generation.ipynb`, before publishing.

**Do not upload the full CelebA image dataset to GitHub.** Keep the dataset local and document how to obtain it instead. Also ensure that any images you commit are appropriate to share and that you have the rights to redistribute them.

## Learning Goals

This project provides practice with:

- PyTorch tensors and `DataLoader`
- Image preprocessing and normalization
- Defining neural networks with `nn.Module`
- Generator and discriminator design
- Binary cross-entropy loss
- Gradient computation with `backward()`
- Parameter updates with Adam
- Alternating optimization in GAN training
- Visualizing generated image batches

## Dataset Acknowledgment

This project uses the CelebA face attributes dataset as its intended source of real training images. Please consult the official CelebA project page for dataset access, citation information, and applicable terms:

- http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

This is an educational implementation and is not affiliated with or endorsed by the CelebA dataset maintainers.
