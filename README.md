# Classification + Domain Adaptation

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
Make sure config to PROD to train and evaluate the whole architecture on (almost) all the dataset.

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
- lambda evolution: $\lambda = \frac {2} {1 + exp(-10 * epoch / maxepoch)} - 1$
<img src="https://latex.codecogs.com/gif.latex?\lambda = \frac {2} {1 + exp(-10 * epoch / maxepoch)} - 1" /> 

## Results
    
| Network       | Source (accuracy) | Target (accuracy) |
| ------------- |:-----------------:| -----------------:|
| CNN           | SVHN (0.91)       | MNIST (0.60)      |
| CNN           | MNIST          | SVHN        |
| CNN-GRL       | SVHN  (0.88)      | MNIST (0.71)      |
| CNN-GRL       | MNIST        | SVHN      |


[cnn]: img/cnn.png "CNN architecture"
[cnn_grl_model]: img/cnn_grl/model.png "CNN-GRL architecture"
[cnn_grl_fe]: img/cnn_grl/feature_extractor.png "CNN-GRL architecture"
