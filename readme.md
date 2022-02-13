## Upper-level policy imitation learning with Pareto-improvements for energy-efficient advanced machining systems

## Training

### Pretrain

```
python main.py --load-ob-rms-only --use-gail 7 --use-critic 1
```

### Finetune

```
python main.py --moo-mode --use-gail 7 --use-critic 2 --trained-model "trained_models/20220213_012544/EP_4000_-12_0.pt" --gail-model  "trained_models/20220213_012544/EP_4000_-12_0_d.pt"
```

## Tensorboard

```
tensorboard --host 0.0.0.0 --logdir logs/runs
```
