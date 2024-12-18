
# Diffusion-models
This repo consist of modules,files,training lopp and inference for the diffusion algorithms like DDPM.

**Objective** : The objective of the this repo is to learn , build and train diffusion models

## DDPM(Denoising Diffusion probabilistic Model)

**Modules**::

**1.Dataset.py**:
- Supports three datasets: MNIST, FashionMNIST, and CIFAR10.
- Downloads and transforms the dataset:
     - Converts images to tensors.
     - Pads grayscale images (MNIST/FashionMNIST) to 32x32.
     - Normalizes images to the range [-1, 1].
- Implements __len__ to get dataset size and __getitem__ to fetch and process individual images.
  
**2.Unet.py**
- Provides a U-Net architecture used in diffusion models for denoising or predicting the noise added to the image at each timestep.
- The U-Net helps process images through downsampling and upsampling paths to capture both global context and fine-grained details, making it effective for learning latent representations and performing accurate conditioning in diffusion models.
  
**3.DiffusionModel.py**
- Base Class:
      - This class extends pl.LightningModule (from PyTorch Lightning), making it compatible with Lightning's training loops, logging, and optimization handling.
- Components:
      - U-Net Backbone:
          - The model uses a U-Net architecture to predict noise at each timestep. The U-Net parameters can be customized (e.g., depth, dimensionality).
      - Beta Scheduling:
          - Implements a linear variance schedule with beta_small and beta_large, defining the noise level for each timestep t.
_key methods_
- forward(x, t): Runs the input image x and timestep t through the U-Net to predict noise.
- beta(t): Computes the noise coefficient (beta) at timestep t using linear interpolation.
- alpha(t) and alpha_bar(t): alpha(t) represents the inverse of beta(t) (used to scale the input image).alpha_bar(t) is the product of alphas up to timestep t, representing the cumulative effect of noise.
- get_loss(batch, batch_idx):
 - Implements the forward diffusion process (Algorithm 1 from Ho et al., 2020).
 - Adds noise to the image batch and computes the Mean Squared Error (MSE) between predicted noise and actual noise.
- denoise_sample(x, t):
 - Implements the reverse denoising process (Algorithm 2 from Ho et al., 2020).
 - Removes noise step-by-step, predicting a cleaner image at each timestep.
- Training and Validation Steps: training_step(batch, batch_idx) and validation_step(batch, batch_idx) calculate the loss and log it during training and validation.
- Optimizer: Uses the Adam optimizer with a learning rate of 2e-4.
  
**4.train.py**
- This contains the Training loop required to build the Model
  
**5.Inference.py**
- This contains the Inference loop required to build the model
  
**6.DDPM_Diffusion.ipynb**
- colab training notebook for the module
  
**Sample Output**
- The output is taken from timestep t=0 to timestep t=1500
- input_image is also provided in the repo
- **At t=0**
  
       ![OutputImaget0](https://github.com/Sabari231024/Diffusion-models/blob/main/DDPM/output/opt0.png?raw=true)

  **At t=50**

       ![OutputImaget0](https://github.com/Sabari231024/Diffusion-models/blob/main/DDPM/output/opt50.png?raw=true)

  **At t=500**

       ![OutputImaget0](https://github.com/Sabari231024/Diffusion-models/blob/main/DDPM/output/opt500.png?raw=true)

  **At t=1000**

       ![OutputImaget0](https://github.com/Sabari231024/Diffusion-models/blob/main/DDPM/output/opt1000.png?raw=true)


  
**Tags:** `Pytorch`, `Diffusion`,'pl-lightning Modelule'

**Author:** Sabari Srinivas  
