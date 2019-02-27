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
    
| Network       | Source (accuracy)       | Target (accuracy) |
| ------------- |:-------------:| -----:|
| CNN           | SVHN (0.91)   | MNIST (0.60) |
| CNN-GRL       | SVHN          | MNIST |
