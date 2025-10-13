\# Adversarial Robustness in Machine Learning



\*\*UE23CS352A Machine Learning Mini-Project\*\*



\## Team Members

\- Sreekar M

\- \[Partner Name]



\## Results

\- \*\*Baseline Model Accuracy\*\*: 43.23%

\- \*\*Adversarial Model Accuracy\*\*: 27.02%



\## How to Run

```bash

pip install -r requirements.txt

python train.py --mode baseline --dataset cifar --epochs 1 --batch\_size 64 --device cpu

python train.py --mode adv --dataset cifar --epochs 1 --batch\_size 64 --device cpu

