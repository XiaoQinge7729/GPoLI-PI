@echo off

python main.py --moo-mode --use-gail 7 --use-critic 2 --trained-model "trained_models/20220212_122358/EP_2005_-6_0.pt" --gail-model  "trained_models/20220212_122358/EP_2005_-6_0_d.pt"

pause