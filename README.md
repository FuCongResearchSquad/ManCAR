# ManCAR

This repository provides a PyTorch reference implementation of the main models and training procedures described in our paper:

> Kun Yang, Yuxuan Zhu, Yazhe Chen, Siyao Zheng, Bangyang Hong, Kangle Wu, Yabo Ni, Anxiang Zeng, Cong Fu, Hui Li.  **ManCAR: Manifold-Constrained Latent Reasoning with Adaptive Test-Time Computation for Sequential Recommendation**.


## Paper & Resources

- Hugging Face Papers:  
- arXiv: 
- Dataset (Hugging Face): 

## Dataset process

you can download CDs dataset from Hugging Face

After downloading the dataset, you need put the dataset into `dataset/processed/`.

or use the following commands to process your datasets

1. Download the dataset from [Amazon](https://amazon-reviews-2023.github.io/)

2. python ./datasets/process_data.py

3. python ./datasets/item_csv.py

After processed, you need to put the processed dataset into `dataset/processed/`.

## Requirements
torch==2.4.1

numpy

tqdm

## Training
To run ManCAR, use the following command:

1. cd ManCAR
2. bash run.sh

## Acknowledgements

We greatly appreciate the official [ReaRec](https://github.com/TangJiakai/ReaRec) repository. Our code is based on the ReaRec repository.
