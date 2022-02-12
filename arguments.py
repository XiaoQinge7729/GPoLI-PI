import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='GPoLI-PI')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--moo-mode', action='store_true', default=False,
                        help='Multi-objective optimization mode')
    parser.add_argument('--trained-model', default='',
                        help='Example: trained_models/xxx/xxx.pt')
    parser.add_argument('--load-ob-rms-only', action='store_true', default=False,
                        help='Load running_mean_std of inputs from saved model, excluding model weights')
    parser.add_argument('--freeze-obs-norm', action='store_true', default=True,
                        help='Freeze obs normalization')
    parser.add_argument('--use-gail', type=int, default=0,
                        help='Gail mode: 0~OFF, 1~Vanilla Gail, 2~Energy-Based, 3~Simple Trajectory, 7~Energy-Based Trajectory (default: 0)')
    parser.add_argument('--gail-model', default='',
                        help='Discriminator model, Example: trained_models/xxx/xxx_d.pt')
    parser.add_argument('--use-critic', type=int, default=0,
                        help='Critic mode: 0~OFF, 1~Standard State, 2~Moo State (default: 0)')
    
    args = parser.parse_args()

    return args
