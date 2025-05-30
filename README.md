# Training Env
## Installation
```bash
pip install -r requirements.txt
mkdir run # for storing running results and hyperparameters
mkdir exp # for storing evaluation results
```

## Train
- single-agent training
```bash
python multitrain.py
```
- multi-agent training
```bash
python MARL_train.py 
```

## Evaluation
```bash
python vrp_baseline.py
```