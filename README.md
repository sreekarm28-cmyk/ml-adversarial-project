# Adversarial Robustness in Machine Learning

**UE23CS352A Machine Learning Mini-Project**

## Team Members
- Sreekar k
- Kaushalya 

## Results

### Baseline Model
- **Clean Accuracy**: 10.00%
- **PGD Undefended**: 9.74%
- **PGD JPEG**: 9.77%
- **PGD Gaussian**: 9.74%
- **PGD K-means**: 9.71%
- **PGD TVM**: 9.94%
- **PGD VQ-VAE**: 10.00%

### Adversarial Model
- **Clean Accuracy**: 10.00%
- **PGD Undefended**: 10.13%
- **PGD JPEG**: 10.08%
- **PGD Gaussian**: 10.13%
- **PGD K-means**: 10.13%
- **PGD TVM**: 10.16%
- **PGD VQ-VAE**: 10.00%

## Observation
The adversarially trained model shows slightly better robustness against PGD attacks compared to the baseline model.

## How to Run
```bash
pip install -r requirements.txt
python train.py --mode baseline --dataset cifar --epochs 1 --batch_size 64 --device cpu
python train.py --mode adv --dataset cifar --epochs 1 --batch_size 64 --device cpu
