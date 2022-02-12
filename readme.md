## Upper-level policy imitation learning with Pareto-improvements for energy-efficient advanced machining systems

## Training

### Pretrain

```
python main_a2c_ppo.py --load-ob-rms-only --use-gail 7 --use-critic 1
```

### Finetune

```
python main_a2c_ppo.py --moo-mode --use-gail 7 --use-critic 2 --trained-model "trained_models/20220212_122358/EP_4500_-5_0.pt" --gail-model  "trained_models/20220212_122358/EP_4500_-5_0_d.pt"
```

## Tensorboard

```
tensorboard --host 0.0.0.0 --logdir logs/runs
```
