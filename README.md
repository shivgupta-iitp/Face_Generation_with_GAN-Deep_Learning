# Face Generation with Generative Adversarial Networks (GAN)

A beginner-friendly deep learning project that uses **PyTorch and Generative Adversarial Networks (GANs)** to generate synthetic face images inspired by the CelebA dataset.

This project demonstrates image preprocessing, neural network architecture, adversarial training, loss functions, and image generation using a Generator and a Discriminator.

## 📌 Project Overview

Generative Adversarial Networks (GANs) consist of two neural networks that compete against each other:

- **Generator (G):** Generates synthetic face images from random noise.
- **Discriminator (D):** Classifies images as real or fake.

During training, the Generator learns to produce more realistic images, while the Discriminator learns to distinguish generated images from real ones.

## 🧠 How GAN Works

```text
Random Noise (100-dimensional vector)
                 |
                 v
            Generator
                 |
                 v
          Generated Face
                 |
                 v
            Discriminator <----- Real Face from Dataset
                 |
                 v
          Real or Fake Prediction
```

The Generator tries to fool the Discriminator, while the Discriminator tries to identify generated images correctly.

## 🏗️ Model Architecture

### 1. Generator Network

The Generator takes a random noise vector of 100 dimensions and transforms it into a 64 × 64 RGB image.

**Architecture:**

- Linear Layer: 100 → 256 + ReLU
- Linear Layer: 256 → 512 + ReLU
- Linear Layer: 512 → 1024 + ReLU
- Linear Layer: 1024 → 12,288 + Tanh
- Reshape output to 3 × 64 × 64

The final `Tanh` activation produces values in the range [-1, 1].

### 2. Discriminator Network

The Discriminator receives an RGB image and predicts whether it is real or generated.

**Architecture:**

- Flatten image: 3 × 64 × 64 → 12,288
- Linear Layer: 12,288 → 1024 + LeakyReLU
- Linear Layer: 1024 → 512 + LeakyReLU
- Linear Layer: 512 → 256 + LeakyReLU
- Linear Layer: 256 → 1 + Sigmoid

The Sigmoid activation produces a score between 0 and 1.

**Note:** This project implements a vanilla GAN using fully connected layers, not a DCGAN.

## 📂 Dataset

This project is designed to use the **CelebA face dataset**.

Dataset directory expected by the notebook:

```text
img_align_celeba/
```

The notebook expects the dataset to be available locally. It does not download the dataset automatically.

### Image Preprocessing

Images are processed using Torchvision transforms:

1. Center crop to 178 × 178.
2. Resize to 64 × 64.
3. Convert images into PyTorch tensors.
4. Normalize RGB channels to approximately [-1, 1].

The notebook reports 202,599 images when the expected full dataset is present.

**Important:** Do not upload the full dataset to GitHub. Follow the dataset's access and usage terms.

## ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow (PIL)
- Matplotlib
- Jupyter Notebook

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install Dependencies

```bash
python -m pip install torch torchvision numpy pillow matplotlib jupyter
```

### 3. Prepare the Dataset

Place the extracted dataset in the project directory:

```text
project/
├── 11 DOWNLOAD GAN CODE.ipynb
├── README.md
└── img_align_celeba/
```

### 4. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `11 DOWNLOAD GAN CODE.ipynb` and execute the cells in order.

## 🔧 Training Configuration

| Parameter | Value |
|---|---|
| Latent Dimension | 100 |
| Image Size | 64 × 64 |
| Image Channels | 3 (RGB) |
| Batch Size | 128 |
| Epochs | 5 |
| Learning Rate | 0.0002 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Adam Betas | (0.5, 0.999) |

The notebook checks for Apple MPS first, then CUDA, and falls back to CPU if neither is available.

Training time depends on the hardware and available system resources.

## 📊 Training Process

The training loop alternates between two steps:

**Discriminator Training**
- Receives real images and generated images.
- Learns to classify real images as 1 and generated images as 0.
- Updates its parameters using backpropagation and Adam.

**Generator Training**
- Generates images from random noise.
- Attempts to make the Discriminator classify those images as real.
- Updates its parameters to improve the generated images.

The notebook logs Generator and Discriminator losses periodically during training.

## 🖼️ Generated Images

After each epoch, the notebook generates and displays a grid of eight synthetic images.

Early outputs may appear noisy or unrealistic. The model needs sufficient training to learn useful visual patterns.

The current notebook displays generated images but does not automatically save them to files.

To save each output grid, add the following line before `plt.show()` inside the image-display function:

```python
plt.savefig(f"generated_epoch_{epoch + 1}.png", bbox_inches="tight")
```

## 📈 Limitations and Future Improvements

This project is an educational implementation and can be improved in several ways:

- Implement a **DCGAN** using convolutional and upsampling layers.
- Train for more epochs and compare generated samples.
- Save model checkpoints to resume training.
- Use fixed random noise to compare image quality across epochs.
- Add random seeds for more reproducible experiments.
- Filter invalid files from the dataset directory.
- Evaluate image quality and diversity using appropriate metrics.

Loss values alone do not guarantee realistic or diverse generated images.

## 📁 Project Structure

```text
.
├── 11 DOWNLOAD GAN CODE.ipynb
├── README.md
└── img_align_celeba/       # Local dataset; do not upload
```

## 🎯 Learning Outcomes

Through this project, I practiced:

- Understanding Generative Adversarial Networks.
- Building neural networks with PyTorch.
- Working with tensors and DataLoaders.
- Preprocessing and normalizing images.
- Implementing Generator and Discriminator networks.
- Understanding Binary Cross Entropy loss.
- Training neural networks using backpropagation.
- Using Adam optimizers.
- Generating and visualizing synthetic images.

## 📚 Dataset Reference

CelebA — Large-scale CelebFaces Attributes Dataset

Official project page: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Please follow the dataset's access, citation, and usage requirements.

---

**Built as a deep learning learning project to explore Generative Adversarial Networks using PyTorch.**
