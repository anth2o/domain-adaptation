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

## Train a specific model

To display training options

```bash
python train.py -h
```

## Load weights and evaluate models

```bash
python evaluate.py
```

## Results
    
| Network       | Source (accuracy) | Target (accuracy) |
| ------------- |:-----------------:| -----------------:|
| CNN           | SVHN (0.908)      | MNIST (0.601)     |
| CNN-GRL       | SVHN  (0.883)     | **MNIST (0.711)**     |
| CNN           | MNIST (0.986)     | SVHN (0.230)      |
| CNN-GRL       | MNIST (0.982)     | SVHN (0.238)      |

It seems than the Gradient Reversal Layer leads to a significative improvement of the network to classify MNIST when it has been trained on SVHN. However, the opposite task isn't more effective with the GRL.

## Features visualization

The next two plots are a visualization of the features built by different networks. It is a t-sne with 2 components of the output of the Dense layer with 512 units in each network. I only used 3000 samples to make t-sne computation faster.

![tsne_cnn_grl]
![tsne_cnn]

The goal of the network with gradient reversal layer is to build features independent of the input distribution samples (MNIST or SVHN for example). Thus, the features built by the CNN-GRL network should be more mixed than the one built by the classic CNN.  
Even if it isn't completely obvious, the features built by the CNN-GRL architecture seem to be more mixed than the ones built by the CNN architecture.  
The same calculation on the whole datasets would probably lead to a more obvious result.



[cnn]: img/model_architecture/cnn.png "CNN architecture"
[cnn_grl_model]: img/model_architecture/cnn_grl/model.png "CNN-GRL architecture"
[cnn_grl_fe]: img/model_architecture/cnn_grl/feature_extractor.png "CNN-GRL architecture"
[tsne_cnn_grl]: img/tsne_features_3000/cnn_grl_train_svhn.png "CNN-GRL features visualization (trained on SVHN)"
[tsne_cnn]: img/tsne_features_3000/cnn_train_svhn.png "CNN features visualization (trained on SVHN)"
[Unsupervised Domain Adaptation by Backpropagation]: https://arxiv.org/pdf/1409.7495.pdf
