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

All the configuration variables are in the file utils/config.py

## Train a specific model

Change model name in train.py (CNN or CNNGRL)

```bash
python train.py
```

## Load weights and evaluate models

```bash
python evaluate.py
```

## Architectures

To have a baseline of the performance without domain adaptation, I tested a simple Convolutional Neural Network:

![cnn]

The network with domain adaptation was designed such that the architecture for label prediction was the same as the previous CNN for two main reasons:
- compare similar networks performance
- load pre trained weigths to make training easier

![cnn_grl_model]
![cnn_grl_fe]

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
