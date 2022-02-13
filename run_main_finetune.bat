@echo off

python main.py --moo-mode --use-gail 7 --use-critic 2 --trained-model "trained_models/20220213_012544/EP_4000_-12_0.pt" --gail-model  "trained_models/20220213_012544/EP_4000_-12_0_d.pt"

pause