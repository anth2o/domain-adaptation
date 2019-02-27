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

## Results

CNN-GRL trained on train set for SVHN (label) and MNIST (no label):
    accuracy on MNIST test set: 0.56

CNN trained on train set for SVHN (label) and MNIST (no label):
    accuracy on MNIST test set: 0.11