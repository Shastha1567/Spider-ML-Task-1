# Spider-ML-Task-1
This Repository contains the club Spider's Machine Learning Task 1
# Spider ML Task 1

Fashion-MNIST classification with a custom neural network in PyTorch, plus a bonus autoencoder.

## Structure

spider_ml_task_1/
├── README.md
├── base_task/
│   ├── notebooks/
│   │   └── spider_ml_task1_base.ipynb
│   ├── saved_models/
│   │   ├── fashion_net_weights.pkl
│   │   └── fashion_net_state_dict.pth
│   ├── submission.csv
│   └── README.md
├── bonus_task/
│   ├── code/
│   │   └── spider_ml_task1_bonus_autoencoder.ipynb
│   ├── results/
│   │   ├── autoencoder_loss_plots.png
│   │   ├── reconstruction_latent_8.png
│   │   ├── reconstruction_latent_16.png
│   │   ├── reconstruction_latent_32.png
│   │   ├── reconstruction_latent_64.png
│   │   ├── latent_dim_comparison.png
│   │   ├── tsne_latent_space.png
│   │   └── denoising_result.png
│   └── README.md

## How to run

Open the notebooks in Google Colab.
Run all cells from top to bottom.
The notebooks download the dataset automatically.

## Requirements

PyTorch, NumPy, Pandas, Matplotlib, Pickle, scikit-learn
All available in Colab by default.
