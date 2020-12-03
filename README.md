# Audio_Separation

Final Project for Duke ECE685D Fall 2020 (Introduction to Deep Learning), "Audio Separation Using Deep Neural Networks"

## Project Goal

The goal of this project is to use nonlinear techniques such as deep learning to separate audio sources from their mixed signal. The mixed signal may be a mixture of music background, dog barking, and siren voice.


## Reproducible Report

- Title: Audio Separation Using Deep Neural Networks

- Reproducible colab notebook: [Audio_Separation_Demo.ipynb](https://colab.research.google.com/drive/1jBL5pusKt0ZcOHcxr8DnVxtNTbzk6o-e?usp=sharing)

- Authors: Jiajun Song & Zhi (Heather) Qiu

- TA: Mohammadreza Soltani



## Methods

- ICA

  - Sklearn implementation for FastICA: [sklearn.decomposition.FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)
  - Example of using FastICA: [Blind source separation using FastICA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html)
  - The nussl library: [Independent Component Analysis using nussl](https://nussl.github.io/docs/examples/factorization/ica.html)


- Open-unmix & Demucs

  - [Open-Unmix for PyTorch](https://github.com/sigsep/open-unmix-pytorch)
  

- *CNN

  - [Audio Classification and Isolation: A Deep NeuralNetwork Approach](https://github.com/ahpvjk/audio-classification-and-isolation)
  

- *Deep Clustering

  - Hershey, John R., et al. “Deep clustering: Discriminative embeddings for segmentation and separation.” 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2016.

  - Wang, Zhong-Qiu, Jonathan Le Roux, and John R. Hershey. “Alternative objective functions for deep clustering.” 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018.
  
  - [Deep clustering using nussl](https://nussl.github.io/docs/examples/deep/deep_clustering.html)
  

  





  
