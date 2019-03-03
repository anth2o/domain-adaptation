# Classification and Domain Adaptation on MNIST and SVHN datasets

## Overview

This repo implements a domain adaptation neural networks. The paper *[Unsupervised Domain Adaptation by Backpropagation]* by Yaroslav Ganin and Victor Lempitsky was a great source of inspiration to design the neural networks.  
The goal of this repo is to implement a network able to classify MNIST samples by using only **SVHN labelled** samples and **MNIST unlabelled** samples during training.

## Depedencies

```bash
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Download data

```bash
python utils/download_data.py
```

## Configuration

All the configuration variables are in the file utils/config.py.  

## Train a specific model

To display training options

```bash
python train.py -h
```

## Load weights and evaluate models

```bash
python evaluate.py
```

## Architectures

To have a baseline of the performance without domain adaptation, I tested a simple Convolutional Neural Network. The architecture is described in this picture: [cnn].

The network with domain adaptation was designed such that the architecture for label prediction was the same as the previous CNN for two main reasons:
- compare similar networks performance
- load pre trained weigths to make training easier

The architecture is described in those pictures: [cnn_grl_model] and [cnn_grl_fe].

## Hyperparameters

- optimizer: SGD(lr=1e-3, momentum=0.9, decay=1e-5)
- lambda evolution:   
<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda&space;=&space;\frac&space;{2}&space;{1&space;&plus;&space;exp(-10&space;*&space;epoch&space;/&space;maxepoch)}&space;-&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda&space;=&space;\frac&space;{2}&space;{1&space;&plus;&space;exp(-10&space;*&space;epoch&space;/&space;maxepoch)}&space;-&space;1" title="\lambda = \frac {2} {1 + exp(-10 * epoch / maxepoch)} - 1" /></a>  

## Results
    
| Network       | Source (accuracy) | Target (accuracy) |
| ------------- |:-----------------:| -----------------:|
| CNN           | SVHN (0.908)      | MNIST (0.601)     |
| CNN-GRL       | SVHN  (0.883)     | **MNIST (0.711)**     |
| CNN           | MNIST (0.986)     | SVHN (0.230)      |
| CNN-GRL       | MNIST (0.982)     | SVHN (0.238)      |


[cnn]: img/model_architecture/cnn.png "CNN architecture"
[cnn_grl_model]: img/model_architecture/cnn_grl/model.png "CNN-GRL architecture"
[cnn_grl_fe]: img/model_architecture/cnn_grl/feature_extractor.png "CNN-GRL architecture"
[Unsupervised Domain Adaptation by Backpropagation]: https://arxiv.org/pdf/1409.7495.pdf
