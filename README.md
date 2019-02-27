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

## Train a specific model

Change model name in train.py (CNN or CNNGRL)

```bash
python train.py
```

## Load weights and evaluate models

```bash
python evaluate.py
```

## Results
    
<<<<<<< HEAD
| Network       | Source (accuracy)       | Target (accuracy) |
| ------------- |:-------------:| -----:|
| CNN           | SVHN (0.91)   | MNIST (0.60) |
| CNN-GRL       | SVHN          | MNIST |
=======
| Network       | Source (accuracy)        | Target (accuracy)|
| ------------- |:-------------:| -----:|
| CNN           | SVHN (0.91)       | MNIST (0.11) |
| CNN-GRL       | SVHN (0.89)         | MNIST (0.56) |
>>>>>>> 37879dd4d30315bff289ce349e1695824824b59f
