import os
import time
import math
import random
from collections import deque, OrderedDict
from sortedcontainers import SortedDict
from copy import deepcopy
from itertools import count

import matplotlib.pyplot as plt
import pygmo as pg

import pybullet_envs
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Normal, Categorical

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torchvision.utils import save_image, make_grid

from ekfac import EKFAC
import gail

from baselines import bench
from baselines.common.vec_env import VecMonitor
from envs import VecNormalize
from multiprocessing_env import SubprocVecEnv

from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from amoorl import make_env_mosrl2 as make_env

#~ torch.set_num_threads(1)

use_cuda = torch.cuda.is_available()
use_cuda = False
device   = torch.device("cuda" if use_cuda else "cpu")

def check_nan(x, name='?', title='???', show=False):
        if torch.isnan(x).any():
            print(f"{title}: {name} is NaN !!!")
            if show:
                print(x)
            return 1
        return 0

def render_moo(pop, hv=0):
    fig, ax = plt.subplots(tight_layout=True)
    pop = np.asarray(pop)
    indices = pg.non_dominated_front_2d(pop[:, :2])
    ndf = pop[indices]
    #~ pg.plot_non_dominated_fronts(pop[:, :2], axes=ax)
    ax.plot(pop[:, 0], pop[:, 1], 'bo')
    ax.plot(ndf[:, 0], ndf[:, 1], 'ms')
    ax.grid('on')
    plt.title('hv {:.3f}, pop {}, ndf {}, {:.0f}%'.format(hv, len(pop), len(ndf), 100*len(ndf)/len(pop)))
    plt.xlabel('SEC')
    plt.ylabel('Tp(s)')
    #~ plt.xlim(14, 20)
    #~ plt.ylim(400, 570)
    return fig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env_gym(env_name, seed, rank):
    def _thunk():
        env = gym.make(env_name)
        env.seed(seed + rank)
        #~ env = bench.Monitor(env, filename=None, allow_early_resets=False)
        return env
    return _thunk

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)
    return None

def get_lr(optimizer):
    lrs = {}
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('layer_rotation', str(i))
        lrs[name] = param_group['lr']
    return lrs

def get_lr_layer_rotation(optimizer):
    lrs = {}
    for i, param_group in enumerate(optimizer.param_groups):
        name = param_group.get('layer_rotation')
        if name:
            lrs[name] = param_group['lr']
    return lrs

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_entropy_schedule(epoch, total_num_epochs, initial_entropy, min_entropy=1e-4):
    if epoch>=total_num_epochs:
        coeff_entropy = min_entropy
    else:
        coeff_entropy = initial_entropy - (initial_entropy * (epoch / float(total_num_epochs)))
    return max(coeff_entropy, min_entropy)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class sActor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=16, z_dim=8):
        super().__init__()
        
        ly1 = (input_size+1), hidden_size
        ly2 = (hidden_size+1), hidden_size
        ly3 = (hidden_size+1), output_size
        
        iw1 = ly1[0]*ly1[1]
        iw2 = ly2[0]*ly2[1] + iw1
        iw3 = ly3[0]*ly3[1] + iw2
        
        self.layers = (0, iw1, ly1, torch.tanh), (iw1, iw2, ly2, torch.tanh), (iw2, iw3, ly3, None)
        
        self.w_dim = iw3
        self.out_dim = iw3
        self.z_dim = z_dim
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.w = None
        self.param_noise = None
        self.z = None
        
#         self.activation = torch.tanh
        self.activation = torch.relu
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))
        
        self.fc1 = init_(nn.Linear(self.z_dim, 32, bias=True))
        self.fc2 = init_(nn.Linear(32, 64, bias=True))
        self.fc3 = init_(nn.Linear(64, 128, bias=True))
        self.fc4 = init_(nn.Linear(128, self.out_dim, bias=True))
        
        print(self)
    
    def __str__(self):
        G = f'G: {self.z_dim}x32 32x64 64x128 128x{self.out_dim}'
        A = f'A: {self.input_size+1}x{self.hidden_size} {self.hidden_size+1}x{self.hidden_size} {self.hidden_size+1}x{self.output_size}'
        return G + '\n' + A

    def decode(self, z):
        x = self.activation(self.fc1(z))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
    def generate(self, z):
        self.z = z
        w = self.decode(z)
        self.w = w[:, :self.w_dim]
        return self.z
    
    def generate_ep(self, z, mask=None):
        self.z = torch.where(mask, z, self.z)
        w = self.decode(z)
        w1 = w[:, :self.w_dim]
        mask = mask.expand_as(w1)
        self.w = torch.where(mask, w1, self.w)
        return self.z
    
    def get_theta(self, i=0):
        return self.w[i].clone().detach()
    
    def set_noise(self, noise=0.01):
        self.param_noise = noise
    
    def forward_layer(self, x, w):
        b = torch.ones(x.size(0), x.size(1), 1)
        x = torch.cat((b, x), dim=-1)
        if self.param_noise:
            x = torch.bmm(x, w + torch.randn_like(w) * self.param_noise)
        else:
            x = torch.bmm(x, w)
        return x
    
    def forward_net(self, x):
        for a,b,s,act in self.layers:
            x = self.forward_layer(x, self.w[:, a:b].reshape(-1, *s))
            if act is not None:
                x = act(x)
        return x
    
    def rbf(self, x1, x2, gamma=0.7):
        alpha = (x1 - x2).pow(2).sum(-1)
        return torch.exp(-gamma * alpha)
    
    def random_proj_reg(self, batch_size, perm=0):
        w = self.w.view(batch_size, -1)
        if perm:
            w = w[torch.randperm(batch_size)]
        x1 = w[:-1, :]
        x2 = w[1:, :]
        return self.rbf(x1, x2).pow(2).mean()
    
    def forward(self, x):
        mean = self.forward_net(x.unsqueeze(1))
        return mean.squeeze(1)

class ActorCritic(nn.Module):
    def __init__(self, input_size, action_space, hidden_size, z_size=16, quaBits=20, 
                            log_std_max=2, log_std_min=-20, limit_std=1, recomb=0, use_critic=1):
        super().__init__()
        
        if action_space.__class__.__name__ == "Discrete":
            output_size = action_space.n
            self.continuous = False
        elif action_space.__class__.__name__ == "Box":
            output_size = action_space.shape[0]
            self.continuous = True
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.z_size = z_size
        
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.limit_std = limit_std
        
        self.quaBits = quaBits
        self.quaMode = 2
        #~ self.quaMode = 1
        #~ self.quaMode = 0
        
        self.quaBits = self.quaBits if self.quaMode else 1
        
        self.use_critic = use_critic
        if use_critic:
            self.critic = Critic0(input_size, hidden_size)
        #~ self.actor = Actor0(input_size, output_size, hidden_size)
        
        self.actor = sActor(input_size, output_size, hidden_size, z_size)
    
    def save(self, t_start, j, reward, reward1, ob_rms=None, opt_state=None):
        save_path = os.path.join("trained_models", time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start)))
        os.makedirs(save_path, exist_ok=True)
        model_name = f"EP_{j}_{reward:.0f}_{reward1:.0f}"
        torch.save({"model":self.state_dict(), "ob_rms":ob_rms, "opt_state":opt_state}, 
                            os.path.join(save_path, model_name + ".pt"))

    def load(self, path, ob_only=False):
        data = torch.load(path)
        if ob_only:
            return data.get('ob_rms'), None
        self.load_state_dict(data["model"])
        return data.get('ob_rms'), data.get('opt_state')
    
    def vis(self):
        vis = []
        if hasattr(self.critic, 'vis'):
            vis.append( self.critic.vis() )
            return torch.cat(vis, dim=-1)
        if hasattr(self.actor, 'vis'):
            vis.append( self.actor.vis() )
            return torch.cat(vis, dim=-1)
        #~ if vis:
            #~ return torch.cat(vis, dim=-1)
        return torch.zeros(1,1,1,1)
    
    def gloss(self):
        loss = torch.tensor(0.0)
        if hasattr(self.critic, 'gloss'):
            loss += self.critic.gloss() * 0.1
        if hasattr(self.actor, 'gloss'):
            loss += self.actor.gloss() * 0.01
        return loss
    
    def generate(self, valid_z=None):
        if valid_z is None:
            z = torch.randn(self.quaBits, self.z_size).to(device)
        else:
            z = valid_z.expand(self.quaBits, self.z_size).to(device)
        self.actor.generate(z)
        #~ z = torch.randn(self.quaBits, self.z_size).to(device)
        #~ self.critic.generate(z)
        return z
    
    def generate_ep(self, valid_z=None, mask=None):
        if valid_z is None:
            z = torch.randn(self.quaBits, self.z_size).to(device)
        else:
            z = valid_z.expand(self.quaBits, self.z_size).to(device)
        self.actor.generate_ep(z, mask=mask)
        return z
    
    def get_theta(self, i=0):
        return self.actor.get_theta(i)
    
    def get_param(self):
        return self.actor.get_param()
    
    def get_param_size(self):
        return self.actor.get_param_size()
    
    def reset(self):
        if hasattr(self.actor, 'reset'):
            self.actor.reset()
    
    def value(self, state):
        return self.critic(state)
    
    def forward(self, state):
        mean = self.actor(state)
        return mean
    
    def sample(self, state, deterministic=False, limit_std=None):
        value = self.critic(state) if self.use_critic else torch.zeros(state.size(0), 1)
        if self.continuous:
            return self.sample_cont(state, deterministic, limit_std=limit_std) + (value,)
        else:
            return self.sample_disc(state, deterministic) + (value,)
    
    def sample_cont(self, state, deterministic=False, limit_std=None):
        mean = self.forward(state)
        mean = torch.tanh(mean)
        if not isinstance(mean, tuple):
            mean = (mean, limit_std if limit_std else self.limit_std)
        dist = Normal(*mean)
        action = dist.sample() if not deterministic else dist.mean.detach()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()
    
    def sample_disc(self, state, deterministic=False):
        mean = self.forward(state)
        #~ mean = torch.sigmoid(mean)
        dist = Categorical(F.softmax(mean, dim=-1))
        action = dist.sample() if not deterministic else dist.probs.argmax(dim=-1, keepdim=False).detach()
        log_prob = dist.log_prob(action)
        return action, log_prob, dist.entropy()
    
    def eval(self, states, actions, limit_std=None):
        """ batch_size = quaBits ! """
        with torch.no_grad():
            return self.eval_grad(states, actions, limit_std)
    
    def eval_grad(self, states, actions, limit_std=None):
        a_log_p = []
        for i in range(states.size(0)//self.quaBits):
            state = states[i*self.quaBits:self.quaBits*(i+1)]
            action = actions[i*self.quaBits:self.quaBits*(i+1)]
            self.generate()
            if self.continuous:
                a_log_p.append(self.eval_cont(state, action, limit_std=limit_std))
            else:
                a_log_p.append(self.eval_disc(state, action))
        return torch.cat(a_log_p)
    
    def eval_cont(self, state, action, limit_std=None):
        mean = self.forward(state)
        mean = torch.tanh(mean)
        if not isinstance(mean, tuple):
            mean = (mean, limit_std if limit_std else self.limit_std)
        dist = Normal(*mean)
        log_prob = dist.log_prob(action)
        return log_prob
    
    # todo: no test
    def eval_disc(self, state, action):
        mean = self.forward(state)
        #~ mean = torch.sigmoid(mean)
        dist = Categorical(F.softmax(mean, dim=-1))
        log_prob = dist.log_prob(action)
        return log_prob


class Critic0(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super().__init__()
        
        self.activation = nn.Tanh
        #~ self.activation = nn.ReLU
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))
        
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            self.activation(),
            init_(nn.Linear(hidden_size, hidden_size)),
            self.activation(),
            init_(nn.Linear(hidden_size, 1)),
        )
    
    def generate(self, *args, **kwargs):
        pass
    
    def forward(self, x):
        return self.critic(x)

class Actor0(nn.Module):
    def __init__(self, input_size, output_size, hidden_size,
                        log_std_max=2, log_std_min=-20, limit_std=1):
        super().__init__()
        
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.limit_std = limit_std
        
        self.activation = nn.Tanh
        #~ self.activation = nn.ReLU
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))
        
        self.actor = nn.Sequential(
            init_(nn.Linear(input_size, hidden_size)),
            self.activation(),
            init_(nn.Linear(hidden_size, hidden_size)),
            self.activation(),
        )
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        self.mean_linear = init_( nn.Linear(hidden_size, output_size) )
        self.log_std_linear = nn.Parameter(torch.zeros(1, output_size))
    
    def generate(self, *args, **kwargs):
        pass
    
    def forward(self, x):
        probs = self.actor(x)
        mean = self.mean_linear(probs)
        log_std = self.log_std_linear.expand_as(mean)
        if self.limit_std:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std.exp()

class ActorCritic0(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, 
                                log_std_max=2, log_std_min=-20, limit_std=1):
        super().__init__()
        
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.limit_std = limit_std
        
        #~ self.activation, self.actfn = nn.Tanh, torch.tanh
        self.activation, self.actfn = nn.ReLU, F.relu
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            np.sqrt(2))
        
        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            self.activation(),
            init_(nn.Linear(hidden_size, hidden_size)),
            self.activation(),
            init_(nn.Linear(hidden_size, 1))
        )
        
        self.actor_mlp1 = init_(nn.Linear(num_inputs, hidden_size))
        self.actor_mlp2 = init_(nn.Linear(hidden_size, hidden_size))
        
        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        self.mean_linear = init_( nn.Linear(hidden_size, num_outputs) )
        self.log_std_linear = nn.Parameter(torch.zeros(1, num_outputs))
        #~ self.log_std_linear = init_( nn.Linear(hidden_size, num_outputs) )
        self.init_layer_rotation()
    
    def save(self, t_start, j, reward, reward1, ob_rms=None, opt_state=None):
        save_path = os.path.join("trained_models", time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start)))
        os.makedirs(save_path, exist_ok=True)
        model_name = f"EP_{j}_{reward:.0f}_{reward1:.0f}"
        torch.save({"model":self.state_dict(), "ob_rms":ob_rms, "opt_state":opt_state}, 
                            os.path.join(save_path, model_name + ".pt"))

    def load(self, path, ob_only=False):
        data = torch.load(path)
        if ob_only:
            return data.get('ob_rms'), None
        self.load_state_dict(data["model"])
        return data.get('ob_rms'), data.get('opt_state')
    
    def layer_rotation_parameters(self):
        params = [
            {'params':self.actor_mlp1.parameters(), 'layer_rotation':'actor_mlp1'},
            {'params':self.actor_mlp2.parameters(), 'layer_rotation':'actor_mlp2'},
            {'params':self.mean_linear.parameters(), 'layer_rotation':'mean_linear'},
            {'params':self.log_std_linear},
            {'params':self.critic.parameters()},
        ]
        return params
    
    def init_layer_rotation(self):
        self.register_layer = ['actor_mlp1', 'actor_mlp2', 'mean_linear']
        self.register_w = []
        self.register_w.append(self.actor_mlp1.weight.data)
        self.register_w.append(self.actor_mlp2.weight.data)
        self.register_w.append(self.mean_linear.weight.data)
        self.initial_w = list(map(lambda w:w.clone().detach(), self.register_w))
    
    def get_similarity(self, x, y):
        d = F.cosine_similarity(x, y)
        return d
    
    def get_layer_rotation(self):
        ds = {}
        for n, p, p0 in zip(self.register_layer, self.register_w, self.initial_w):
            d = F.cosine_similarity(p.detach().view(1,-1), p0.view(1,-1))
            ds[n+'_cos'] = 1.0-d.item()
        return ds
    
    def compute_layer_rotation_loss(self):
        loss = []
        for n, p, p0 in zip(self.register_layer, self.register_w, self.initial_w):
            d = self.get_similarity(p.clone().view(1,-1), p0.view(1,-1))
            loss.append(d)
        return torch.cat(loss)
    
    def compute_layer_rotation_lr(self):
        ds = []
        for n, p, p0 in zip(self.register_layer, self.register_w, self.initial_w):
            d = self.get_similarity(p.detach().view(1,-1), p0.view(1,-1))
            ds.append(d)
        return F.softmax(torch.cat(ds), dim=0).numpy()
    
    def set_layer_rotation_lr(self, param_groups, base_lr, alpha=0.5):
        lr_list = self.compute_layer_rotation_lr()
        lr_list = lr_list - lr_list.min() *alpha
        lr_list = lr_list/lr_list.max() *base_lr
        lr_list = lr_list.clip(1e-5, base_lr)
        for param_group, lr in zip(param_groups, lr_list):
            param_group['lr'] = lr
    
    def generate(self, valid_z=None):
        if valid_z is None:
            z = torch.randn(24, 2).to(device)
        else:
            z = valid_z.expand(24, 2).to(device)
        return z
    
    def generate_ep(self, valid_z=None, mask=None):
        if valid_z is None:
            z = torch.randn(24, 2).to(device)
        else:
            z = valid_z.expand(24, 2).to(device)
        return z
    
    def gloss(self):
        return torch.tensor(0.0)
    
    def value(self, state):
        return self.critic(state)
    
    def actor(self, x):
        x = self.actfn(self.actor_mlp1(x))
        x = self.actfn(self.actor_mlp2(x))
        return x
    
    def forward(self, x):
        probs = self.actor(x)
        mean = self.mean_linear(probs)
        log_std = self.log_std_linear.expand_as(mean)
        #~ log_std = self.log_std_linear(probs)
        if self.limit_std:
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
            #~ log_std = torch.tanh(log_std)
            #~ log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mean, log_std
    
    def sample(self, state, deterministic=False, limit_std=None):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        #~ x_t = normal.sample()  # for reparameterization trick (mean + std * N(0,1))
        #~ x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #~ action = torch.tanh(x_t)
        #~ action = x_t.detach()
        #~ log_prob = normal.log_prob(x_t)
        action = normal.sample() if not deterministic else normal.mean.detach()
        log_prob = normal.log_prob(action)
        return action, log_prob, normal.entropy(), self.critic(state)
    
    def eval(self, states, actions, limit_std=None):
        with torch.no_grad():
            mean, log_std = self.forward(states)
            std = log_std.exp()
            normal = Normal(mean, std)
            log_prob = normal.log_prob(actions)
            return log_prob


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def feed_forward_generator(obs_next, obs, actions, num_inputs, output_size, mini_batch_size=64):
    obs_next = torch.cat(obs_next)#.view(-1, num_inputs)
    obs = torch.cat(obs)#.view(-1, num_inputs)
    actions = torch.cat(actions)#.view(-1, output_size)
    batch_length = obs.shape[0]
    sampler = BatchSampler(
        SubsetRandomSampler(range(batch_length)),
        mini_batch_size,
        drop_last=True)
    for indices in sampler:
        obs_next_batch = obs_next[indices]
        obs_batch = obs[indices]
        actions_batch = actions[indices]
        yield obs_batch, actions_batch, obs_next_batch


def main():
    writer = SummaryWriter(os.path.join("logs", 'runs'))
    tsbx_id = time.strftime('%H%M%S_', time.localtime())
    norm_obs = False
    norm_ret = False
    norm_pg = 0
    
    render_env = 10
    render_env = 0
    #~ log_debug = 0
    log_debug = 1
    #~ log_grads = 0
    log_grads = 1
    log_interval = 10
    
    #~ num_envs = 32
    num_envs = 24
    #~ num_envs = 20
    #~ num_envs = 16
    #~ num_envs = 2
    #~ num_envs = 1
    num_steps   = 5
    #~ num_steps   = 10
    num_steps   = 50
    
    ep_valid = 1000
    #~ ep_valid = 500
    #~ ep_max = 10
    #~ ep_max = 24
    #~ ep_max = 48
    #~ ep_max = 50      # train moo
    ep_max = 100        # train normal
    #~ ep_max = 120        # train
    ep_max = 144        # train traj
    
    moo_mode = 0
    #~ moo_mode = 1
    
    env_name = "MOO"
    
    SEED = 567
    #~ SEED = 1234
    #~ SEED = 66666
    #~ SEED = 2020
    #~ SEED = 9700
    #~ SEED = 888
    #~ SEED = 112233
    #~ SEED = 445566
    SEED = 778899
    #~ SEED = 168442
    #~ SEED = 308086
    
    #~ SEED = 42
    #~ SEED = 275
    #~ SEED = 5092
    #~ SEED = 486200
    #~ SEED = 55444709
    
    save_interval = 50
    save_interval = 100
    #~ save_interval = 200
    save_interval = 500
    #~ save_interval = 1e6
    trained_model = None
    trained_model = "trained_models/20210204_231637_case1/EP_6000_-1_0.pt"   # case1 best
    
    ob_rms_only = True
    #~ ob_rms_only = False
    
    finetune_mode = 0
    finetune_mode = 1       # freeze obs_norm
    
    valid_mode = 0
    #~ valid_mode = 2               # by hand
    #~ valid_mode = 1               # by train_valid
    
    train_valid = 0
    train_valid = 100
    
    valid_ep = 5
    if valid_mode: train_valid = 0
    if train_valid: ep_valid = 200
    
    valid_z = None
    #~ valid_z = torch.load("trained_z/20200320_203133.pt")['zs'][0]
    
    render_env = 1 if valid_mode else render_env
    #~ num_updates = 100000
    num_updates = int(1e8)//num_steps//num_envs
    
    #~ latent_size = 1
    latent_size = 2
    #~ latent_size = 3
    #~ latent_size = 4
    #~ latent_size = 8     # ok
    #~ latent_size = 10    # moo
    #~ latent_size = 16
    #~ latent_size = 32
    hidden_size = 16
    hidden_size = 20
    #~ hidden_size = 32
    #~ latent_size = 64
    #~ hidden_size = 64
    
    rnn_enc_size = 10
    #~ rnn_enc_size = 20
    rnn_enc_size = 30
    #~ rnn_enc_size = 50
    #~ rnn_enc_size = 100
    
    rnn_enc_lr = 5e-4
    rnn_enc_lr = 3e-4
    #~ rnn_enc_lr = 1e-4
    
    use_gail = 0
    #~ use_gail = 1
    #~ use_gail = 2            # real traj mode
    use_gail = 3            # simple traj mode
    #~ use_gail = 4        # rnn encoder mode
    #~ use_gail = 5        # triplet_loss
    #~ use_gail = 6        # rnd
    #~ use_gail = 7        # energy traj, work nice
    
    gail_moo_hack = 0       # replace sec
    #~ gail_moo_hack = 1       # replace tp
    
    policy_grad_pen = False
    #~ policy_grad_pen = True
    
    gail_train = True
    #~ gail_train = False
    
    gail_energy = True
    gail_energy = False
    
    if valid_mode==2: use_gail = 0
    if use_gail in (3,4,5,6): gail_energy = False
    if use_gail in (2,7): gail_energy = True
    if use_gail not in (1,3): policy_grad_pen = False
    
    gail_lr = 1e-3
    gail_lr = 5e-4
    #~ gail_lr = 1e-4
    #~ gail_lr = 3e-4
    
    gail_ep = 5
    #~ gail_ep = 10
    gail_warmup = 100
    gail_batch_size = 64
    #~ gail_batch_size = 72    # SingleTimestepIRL # energy
    #~ gail_batch_size = 48    # SingleTimestepIRL # energy
    #~ gail_batch_size = 24    # TrajectoryIRL # energy
    if use_gail in (2,3,4,7) or (use_gail==1 and gail_energy==1):
        gail_batch_size = 48
        #~ gail_batch_size = 24
        #~ gail_warmup = 1
    gail_experts_dir = "gail_experts"
    gail_traj_num = 50
    gail_traj_len = 5
    gail_traj_freq = 1
    gail_experts_trajs, gail_traj_num = "case1_full.pt", 318
    
    gail_model = None
    #~ gail_model = "trained_models/20210928_072012/EP_2500_-8_0_d.pt"
    
    gail_model2 = None
    #~ gail_model = "trained_models/20210219_183043/EP_3000_-10_0_d.pt"
    
    rnd_model = 'rnd_case1_full.pt'
    #~ rnd_model = 'rnd_case1_sec.pt'
    
    GM = 1
    #~ GM = 0
    
    quann_recomb = 0
    #~ quann_recomb = 1
    
    norm_obs = True
    #~ norm_ret = True
    
    scale_ret = 0.01        # Pendulum-v0
    scale_ret = 0.1         # Moo
    #~ scale_ret = 0           # DeepSea
    
    scale_hv = 0.01     # DeepSea
    scale_hv = 0.001    # DeepseaEnergy
    #~ scale_hv = 0.0001    # Puddle
    
    norm_hv = 0
    #~ norm_hv = 1
    
    kfac_freq = 100
    #~ kfac_freq = 50
    #~ kfac_freq = 1
    
    use_kfac = 1
    #~ use_kfac = 0
    
    use_critic = 1
    #~ use_critic = 0
    
    rand_orth_reg = 0
    #~ rand_orth_reg = 1            # train
    rand_orth_reg = 2       # show only
    
    rand_orth_reg = rand_orth_reg if GM else 0
    
    coeff_orth = 0.1        # ok
    
    param_noise = None
    
    limit_std = 1.5         # Pendulum-v0
    limit_std = 0.1         # ok
    
    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -20
    
    coef_rotation_loss = 0.1
    #~ coef_rotation_loss = 0.5
    
    use_layer_rotation = 0
    #~ use_layer_rotation = 1
    #~ use_layer_rotation = 2
    #~ use_layer_rotation = 3
    
    use_scheduler_lr = 0
    #~ use_scheduler_lr = 1
    #~ use_scheduler_lr = 2
    
    cyc_step_size_up = 100
    num_scheduler_lr = 1000     # Pendulum
    max_lr = 4e-3
    
    lr          = 5e-4
    
    max_grad = 0.5
    
    betas = 0.0, 0.9         # kfac
    
    gamma = 0.99
    coef_value = 0.5
    
    coef_entropy = 2
    #~ coef_entropy = None
    initial_entropy = coef_entropy
    min_entropy = 0.1
    
    use_entropy_decay = int(1e4)
    use_entropy_decay = 0
    
    if env_name=='MOO':
        envs = [make_env(env_name, SEED, i, env_args={'debug':render_env}) for i in range(num_envs)]
    else:
        envs = [make_env_gym(env_name, SEED, i) for i in range(num_envs)]
    envs = SubprocVecEnv(envs, render=render_env) # 8 env
    
    envs = VecMonitor(envs)
    envs = VecNormalize(envs, ob=norm_obs, ret=norm_ret, gamma=gamma)

    num_inputs  = envs.observation_space.shape[0]
    action_space = envs.action_space
    if action_space.__class__.__name__ == "Discrete":
        output_size = action_space.n
    elif action_space.__class__.__name__ == "Box":
        output_size = action_space.shape[0]
    
    print("num_inputs", num_inputs)
    print("output_size", output_size)
    
    if GM:
        model = ActorCritic(num_inputs, action_space, hidden_size, latent_size, quaBits=num_envs,
                log_std_max=LOG_SIG_MAX, log_std_min=LOG_SIG_MIN, limit_std=limit_std,
                recomb=quann_recomb, use_critic=use_critic).to(device)
    else:
        num_outputs = envs.action_space.shape[0]
        model = ActorCritic0(num_inputs, num_outputs, hidden_size, 
                log_std_max=LOG_SIG_MAX, log_std_min=LOG_SIG_MIN, limit_std=limit_std).to(device)
    
    def value_fn(s):
        with torch.no_grad():
            return model.value(s)
    
    if use_gail:
        assert len(envs.observation_space.shape) == 1
        if use_gail in (1,5):
            gail_shape = num_inputs, output_size
        elif use_gail==2:
            gail_shape = num_inputs, output_size
        elif use_gail==3:
            gail_shape = num_inputs*gail_traj_len, output_size*gail_traj_len
        elif use_gail==4:
            gail_shape = num_inputs, output_size
            rnn_encoder = gail.RnnEncoder(num_inputs, output_size, rnn_enc_size, lr=rnn_enc_lr).to(device)
        
        if use_gail==6:
            #~ gail_shape = num_inputs, output_size
            gail_shape = num_inputs*gail_traj_len, output_size*gail_traj_len
            discr = gail.RND_Critic(*gail_shape, device)
            discr.load(os.path.join(gail_experts_dir, rnd_model))
        elif use_gail==4:
            discr = gail.Discriminator(*gail_shape, 100, device, lr=gail_lr, input_emb=rnn_enc_size)
        elif use_gail==7:
            gail_shape = num_inputs, output_size
            traj_shape = num_inputs*gail_traj_len + output_size*gail_traj_len
            discr = gail.Discriminator(*gail_shape, 100, device, lr=gail_lr, input_emb=traj_shape)
        else:
            discr = gail.Discriminator(*gail_shape, 100, device, lr=gail_lr)
        file_name = os.path.join(gail_experts_dir, gail_experts_trajs)
        
        if use_gail in (1,5):
            expert_dataset = gail.ExpertDataset(
                file_name, num_trajectories=gail_traj_num, subsample_frequency=gail_traj_freq)
        elif use_gail in (2,3,6,7):
            expert_dataset = gail.ExpertDatasetTraj(
                file_name, num_trajectories=gail_traj_num, subsample_frequency=gail_traj_freq)
        elif use_gail==4:
            expert_dataset = gail.ExpertDatasetRnn(
                file_name, num_trajectories=gail_traj_num, subsample_frequency=gail_traj_freq)
        
        drop_last = len(expert_dataset) > gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=gail_batch_size,
            shuffle=True,
            drop_last=True)
        
        if gail_model:
            discr.load(gail_model)
        
        if gail_model2:
            discr2 = gail.Discriminator(*gail_shape, 100, device, lr=gail_lr)
            discr2.load(gail_model2)
    
    # load model begin
    opt_state = None
    if trained_model is not None:
        ob_rms, opt_state = model.load(trained_model, ob_only=ob_rms_only)
        vec_norm = get_vec_normalize(envs)
        if ob_rms is not None and vec_norm is not None:
            if valid_mode or finetune_mode==1:
                vec_norm.eval()
            vec_norm.ob_rms = ob_rms
    # load model end
    
    if use_kfac:
        tsbx_id += 'kfac_'
        preconditioner = EKFAC(model, 0.1, ra=True, update_freq=kfac_freq)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)

    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    if use_scheduler_lr==2:
        scheduler_lr = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                        base_lr=lr, max_lr=max_lr,
                                                        step_size_up=cyc_step_size_up,
                                                        cycle_momentum=False,
                                                        )
    
    frame_idx    = 0
    test_rewards = []
    
    #~ episode_rewards = torch.zeros([num_envs, 1]).to(device)
    #~ final_rewards = torch.zeros([num_envs, 1]).to(device)
    episode_rewards = deque(maxlen=num_envs)

    best_z = SortedDict()
    env0_rws = 0
    state = envs.reset()
    
    #~ moo_pop = []
    ref_p = None
    mask_w = None
    
    start = time.time()
    
    for j in range(num_updates):
        obs_next = []
        obs = []
        actions = []
        log_probs = []
        entropys = []
        values    = []
        rewards   = []
        masks     = []
        loss_reg = []
        
        moo_pop = []
        moo_z = []
        moo_x = []
        mask_w = None
        hv_reward_idx = []
        ep_reward_idx = []
        ep_start_idx = []
        ep_start = np.zeros(num_envs, dtype=int)
        ep_counts = 0
        ep_in_bounds = {}
        ep_theta = {}
        ep_mobj = {}
        
        gail_moo_map = {}
        gail_moo_pop = []
        
        nan_check = 0
        
        if train_valid:
            valid_mode = 1 if max(train_valid-1, j-1)%train_valid==0 else 0
        
        # generation begin
        if param_noise and not valid_mode:
            model.actor.set_noise(param_noise)
        z = model.generate(valid_z=valid_z if valid_mode else None)
        # generation end
        # rollout trajectory
        state = envs.reset()
        #~ for steps in range(num_steps):
        for steps in count(0):
            state = torch.FloatTensor(state).to(device)
            
            # generation begin
            #~ z = model.generate(valid_z=valid_z if valid_mode else None)
            if mask_w is not None:
                z = model.generate_ep(valid_z=valid_z if valid_mode else None, mask=mask_w)
                if rand_orth_reg:
                    lossReg = model.actor.random_proj_reg(num_envs, perm=1)
                    loss_reg.append(lossReg)
            # generation end
            
            action, log_prob, dist_entropy, value = model.sample(state, deterministic=valid_mode, 
                                                                    limit_std=(coef_entropy if use_entropy_decay else None),
                                                                    )
            
            # debug begin
            nan_check += check_nan(state, 'state', 'main()')
            nan_check += check_nan(action, 'action', 'main()')
            # debug end
            
            next_state, reward, done, infos = envs.step(action.cpu().numpy())
            
            reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
            mask = torch.FloatTensor(1 - done).unsqueeze(1).to(device)
            
            if scale_ret and not norm_ret:
                reward *= scale_ret
            
            if done.any():
                mask_w = torch.BoolTensor(done).unsqueeze(1).to(device)
            else:
                mask_w = None
            
            for i, info in enumerate(infos):
                last_ep_key = None
                if 'episode' in info.keys():
                    r = info['episode']['r']
                    episode_rewards.append(r)
                    ep_reward_idx.append(steps*num_envs + i)
                    ep_start_idx.append([i, ep_start[i], steps+1-ep_start[i]])
                    last_ep_key = str((i, ep_start[i]))
                    ep_in_bounds[last_ep_key] = True
                    ep_start[i] = steps + 1
                    if valid_mode:
                        best_z[r] = z[i]
                        if i==0:
                            env0_rws = r
                if 'mobj' in info.keys() and (moo_mode or valid_mode):
                    moo_pop.append(info['mobj'])
                    moo_z.append(z[i].cpu().numpy())
                    hv_reward_idx.append(steps*num_envs + i)
                    gail_moo_map[steps*num_envs + i] = len(gail_moo_pop)
                    gail_moo_pop.append(info['mobj'].copy())
                if 'mobj_ill' in info.keys():
                    ep_in_bounds[last_ep_key] = False
                if 'ref_p' in info.keys():
                    ref_p = info['ref_p']
                if 'moo_x' in info.keys():
                    moo_x.append(info['moo_x'])
                if valid_mode:
                    ep_theta[last_ep_key] = model.get_theta(i).cpu().numpy()
                    ep_mobj[last_ep_key] = info.get('mobj', np.array([0.,0.]))
            
            log_probs.append(log_prob)
            entropys.append(dist_entropy)
            values.append(value)
            rewards.append(reward)
            masks.append(mask)
            if use_gail or valid_mode==2:
                obs_next.append(torch.FloatTensor(next_state).to(device))
                obs.append(state.clone())
                actions.append(action.clone())
            
            state = next_state
            frame_idx += 1
            
            ep_counts += np.sum(done.astype(np.bool))
            if ep_counts>=(ep_valid if valid_mode else ep_max):
                break
        
        # gail begin
        gail_rewards = []
        gail_rws = []
        gail_r = torch.zeros(num_envs, 1)
        rollouts = None
        rollout_dataset = None
        # hack to save begin
        if valid_mode==2:
            rollout_dataset = gail.RolloutDataset(obs, actions, obs_next, ep_start_idx, ep_in_bounds, ep_theta, ep_mobj, traj_len=gail_traj_len)
        # hack to save end
        if use_gail:
            if j >= 10 and 0:
                envs.venv.eval()

            gail_epoch = gail_ep
            if j < 10:
                gail_epoch = gail_warmup  # Warm up 100
            
            # hack to save rollout for mode 1
            if use_gail in (2,3,6,7) or (use_gail==1 and train_valid and valid_mode):
                rollout_dataset = gail.RolloutDataset(obs, actions, obs_next, ep_start_idx, ep_in_bounds, ep_theta, ep_mobj, traj_len=gail_traj_len)
                rollouts = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    batch_size=gail_batch_size,
                                    shuffle=True,
                                    drop_last=True)
            elif use_gail==4:
                rollout_dataset = gail.RolloutDatasetRnnPack(obs, actions, ep_start_idx, batch_first=True)
                rollouts = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    collate_fn=rollout_dataset.collate_fn_padd,
                                    batch_size=gail_batch_size,
                                    shuffle=True,
                                    drop_last=True)
            
            # Train
            if gail_train and not valid_mode and use_gail!=6:
                gail_loss = 0
                log_p_policy = 0
                log_p_expert = 0
                log_q_policy = 0
                log_q_expert = 0
                d_policy = 0
                d_expert = 0
                d_cosine = 0
                d_mmd = 0
                
                for _ in range(gail_epoch):
                    if use_gail==1:
                        rollouts = feed_forward_generator(obs_next, obs, actions, num_inputs, output_size, gail_batch_size)
                    
                    if use_gail==2:
                        gail_info  = discr.update_energy_traj(gail_train_loader, rollouts, policy=model, value_fn=value_fn)
                    elif use_gail==7:
                        gail_info  = discr.update_energy_traj7(gail_train_loader, rollouts, policy=model, value_fn=value_fn)
                    elif use_gail==4:
                        #~ rnn_encoder.prepare()
                        gail_info  = discr.update_rnn(rnn_encoder, gail_train_loader, rollouts)
                        #~ rnn_encoder.update()
                    elif use_gail==5:
                        rollouts = feed_forward_generator(obs_next, obs, actions, num_inputs, output_size, gail_batch_size)
                        rollouts1 = feed_forward_generator(obs_next, obs, actions, num_inputs, output_size, gail_batch_size)
                        gail_info = discr.update_triplet(gail_train_loader, rollouts, rollouts1)
                    elif not gail_energy:
                        gail_info = discr.update(gail_train_loader, rollouts, obsfilt=None)
                    else:
                        if use_gail==1:
                            gail_info  = discr.update_energy(gail_train_loader, rollouts, policy=model, value_fn=value_fn)
                    
                    gail_loss += gail_info.get('loss', 0)
                    log_p_policy += gail_info.get('log_p_policy', 0)
                    log_p_expert += gail_info.get('log_p_expert', 0)
                    log_q_policy += gail_info.get('log_q_policy', 0)
                    log_q_expert += gail_info.get('log_q_expert', 0)
                    d_policy += gail_info.get('d_policy', 0)
                    d_expert += gail_info.get('d_expert', 0)
                    d_cosine += gail_info.get('d_cosine', 0)
                    d_mmd += gail_info.get('d_mmd', 0)
                
                writer.add_scalar(tsbx_id+'_gail/loss', gail_loss/gail_epoch, j)
                if gail_energy:
                    writer.add_scalars(tsbx_id+'_gail/log_p', {'policy': log_p_policy/gail_epoch, 
                                            'expert': log_p_expert/gail_epoch}, j)
                    writer.add_scalars(tsbx_id+'_gail/log_q', {'policy': log_q_policy/gail_epoch, 
                                            'expert': log_q_expert/gail_epoch}, j)
                    writer.add_scalars(tsbx_id+'_gail/D', {'policy': d_policy/gail_epoch,
                                            'expert': d_expert/gail_epoch}, j)
                elif use_gail in (1,5,3,4):
                    writer.add_scalars(tsbx_id+'_gail/D', {'policy': d_policy/gail_epoch, 
                                            'expert': d_expert/gail_epoch}, j)
                writer.add_scalar(tsbx_id+'_gail/d_cosine', d_cosine/gail_epoch, j)
                writer.add_scalar(tsbx_id+'_gail/d_mmd', d_mmd/gail_epoch, j)
            
            # Predict
            if use_gail in (1,5,2,40):
                gail_rws_ep = []
                rnn_hxs = torch.zeros(num_envs, rnn_enc_size)
                for step in range(steps+1):
                    if use_gail==6:
                        r_gail = discr.get_reward(obs[step], actions[step])
                    
                    elif gail_train:
                        if use_gail==4:
                            r_gail, rnn_hxs = discr.predict_reward_rnn(
                                            rnn_encoder, rnn_hxs,
                                            obs[step], actions[step], 
                                            gamma, masks[step]
                                            )
                        elif gail_energy:
                            r_gail = discr.predict_energy(
                                            obs[step], actions[step], log_probs[step], 
                                            obs_next[step], value_fn,
                                            gamma, masks[step]
                                            ) \
                                            if use_gail==1 else \
                                        discr.predict_energy_traj(obs[step], actions[step],
                                                                                gamma, masks[step])
                        else:
                            r_gail = discr.predict_reward(
                                            obs[step], actions[step], 
                                            gamma, masks[step]
                                            ) \
                                            if not moo_mode else \
                                        discr.predict_reward_raw(
                                            obs[step], actions[step], 
                                            gamma, masks[step])
                    else:
                        if use_gail==4:
                            r_gail, rnn_hxs = discr.predict_reward_rnn(
                                            rnn_encoder, rnn_hxs,
                                            obs[step], actions[step], 
                                            gamma, masks[step],
                                            #~ update_rms=False
                                            ) 
                        elif gail_energy:
                            r_gail = discr.predict_energy_r(
                            #~ r_gail = discr.predict_energy(
                                            obs[step], actions[step], log_probs[step], 
                                            obs_next[step], value_fn,
                                            gamma, masks[step], 
                                            #~ update_rms=False
                                            ) \
                                            if use_gail==1 else \
                                        discr.predict_energy_traj(obs[step], actions[step],
                                                                                gamma, masks[step])
                        else:
                            r_gail = discr.predict_reward(
                                            obs[step], actions[step], 
                                            gamma, masks[step],
                                            #~ update_rms=False
                                            ) \
                                            if not moo_mode else \
                                        discr.predict_reward_raw(
                                            obs[step], actions[step], 
                                            gamma, masks[step])
                    # debug begin
                    nan_check += check_nan(r_gail, 'r_gail', 'use_gail')
                    # debug end
                    gail_r += r_gail
                    gail_rws.append(gail_r.clone())   # no use
                    if torch.any(masks[step]==0):
                        gail_rws_ep.append(gail_r[masks[step]==0])
                    # gail_moo_pop hack begin
                    if moo_mode:
                        for i_, m_ in enumerate(masks[step]):
                            if m_!=0:
                                continue
                            mobj_ = gail_moo_map.get(step*num_envs + i_, None)
                            if mobj_ is not None:
                                if gail_moo_hack==0:
                                    gail_moo_pop[mobj_][0] = min(-gail_r[i_], 11)
                                else:
                                    gail_moo_pop[mobj_][1] = min(-gail_r[i_], 11)
                    # gail moo hack reward begin
                    #~ rewards[step] += r_gail
                    # gail moo hack reward end
                    # gail_moo_pop hack end
                    gail_r *= masks[step]
                    gail_rewards.append(r_gail.clone())
                
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
                writer.add_scalar(tsbx_id+'_gail_r/reward_avg', torch.mean(torch.cat(gail_rewards)), j)
            
            elif use_gail==3:
                gail_rws_ep = []
                gail_rewards = torch.zeros(num_envs, steps+1)
                #~ gail_rewards = torch.full((num_envs, steps+1), -1.0)
                rollouts_pred = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
                for traj, traj_idx in zip(rollouts_pred, ep_start_idx):
                    r_gail = discr.predict_reward_traj3(traj[0], traj[1])
                    traj_i, traj_start, traj_steps = traj_idx
                    gail_rewards[traj_i, traj_start:traj_start+traj_steps] = r_gail
                    gail_rws_ep.append(r_gail)
                    # gail_moo_pop hack begin
                    if moo_mode:
                        mobj_ = gail_moo_map.get((traj_start+traj_steps-1)*num_envs + traj_i, None)
                        if mobj_ is not None:
                            if gail_moo_hack==0:
                                gail_moo_pop[mobj_][0] = min(-r_gail, 11)
                            else:
                                gail_moo_pop[mobj_][1] = min(-r_gail, 11)
                    # gail_moo_pop hack end
                gail_rewards = gail_rewards.chunk(steps+1, dim=1)
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
            
            elif use_gail==20:
                gail_rws_ep = []
                gail_rewards = torch.zeros(num_envs, steps+1)
                #~ gail_rewards = torch.full((num_envs, steps+1), -1.0)
                rollouts_pred = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
                for traj, traj_idx in zip(rollouts_pred, ep_start_idx):
                    r_gail = discr.predict_energy_traj2(traj[0], traj[1])
                    traj_i, traj_start, traj_steps = traj_idx
                    gail_rewards[traj_i, traj_start:traj_start+traj_steps] = r_gail
                    gail_rws_ep.append(r_gail)
                gail_rewards = gail_rewards.chunk(steps+1, dim=1)
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
            
            elif use_gail==7:
                gail_rws_ep = []
                gail_rewards = torch.zeros(num_envs, steps+1)
                #~ gail_rewards = torch.full((num_envs, steps+1), -1.0)
                rollouts_pred = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
                for traj, traj_idx in zip(rollouts_pred, ep_start_idx):
                    r_gail = discr.predict_reward_traj7(traj[0], traj[1])
                    traj_i, traj_start, traj_steps = traj_idx
                    gail_rewards[traj_i, traj_start:traj_start+traj_steps] = r_gail
                    gail_rws_ep.append(r_gail)
                    # gail_moo_pop hack begin
                    if moo_mode:
                        mobj_ = gail_moo_map.get((traj_start+traj_steps-1)*num_envs + traj_i, None)
                        if mobj_ is not None:
                            if gail_moo_hack==0:
                                gail_moo_pop[mobj_][0] = min(-r_gail, 11)
                            else:
                                gail_moo_pop[mobj_][1] = min(-r_gail, 11)
                    # gail_moo_pop hack end
                gail_rewards = gail_rewards.chunk(steps+1, dim=1)
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
            
            elif use_gail==6:
                gail_rws_ep = []
                gail_rewards = torch.zeros(num_envs, steps+1)
                #~ gail_rewards = torch.full((num_envs, steps+1), -1.0)
                rollouts_pred = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
                for traj, traj_idx in zip(rollouts_pred, ep_start_idx):
                    r_gail = discr.get_reward(traj[0], traj[1])
                    traj_i, traj_start, traj_steps = traj_idx
                    gail_rewards[traj_i, traj_start:traj_start+traj_steps] = r_gail
                    gail_rws_ep.append(r_gail)
                gail_rewards = gail_rewards.chunk(steps+1, dim=1)
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
            
            elif use_gail==4:
                gail_rws_ep = []
                gail_rewards = torch.zeros(num_envs, steps+1)
                rollouts_pred = torch.utils.data.DataLoader(
                                    dataset=rollout_dataset,
                                    collate_fn=rollout_dataset.collate_fn_padd,
                                    batch_size=1,
                                    shuffle=False,
                                    drop_last=False)
                for traj, traj_idx in zip(rollouts_pred, ep_start_idx):
                    r_gail = discr.predict_reward_rnn4(rnn_encoder, traj[0])
                    traj_i, traj_start, traj_steps = traj_idx
                    gail_rewards[traj_i, traj_start:traj_start+traj_steps] = r_gail
                    gail_rws_ep.append(r_gail)
                gail_rewards = gail_rewards.chunk(steps+1, dim=1)
                gail_rws_ep = torch.cat(gail_rws_ep)
                writer.add_scalars(tsbx_id+'_gail_r/reward', {'mean':gail_rws_ep.mean(), 
                                    'min':gail_rws_ep.min(), 'max':gail_rws_ep.max()}, j)
            
            if not moo_mode:
                rewards = gail_rewards
        # gail end
        
        # hv begin
        if len(gail_moo_pop)>1 and moo_mode and use_gail:
            hv = pg.hypervolume(gail_moo_pop)
            #~ ref_p = [1, 51, 1]              # DeepseaEnergy
            #~ ref_p = [1, 51]              # DeapSea
            #~ ref_p = [41, 1001]              # Moo
            #~ ref_p = hv.refpoint(offset=0.1)
            if gail_moo_hack==0:
                ref_p = [11, 1001]      # [41, 1001]
                hv_scale = 1e-3
            else:
                ref_p = [41, 11]      # [41, 1001]
                hv_scale = 1e-2
            hv_c = hv.contributions(ref_p)
            hv_s = hv.compute(ref_p)
            #~ episode_rewards.append(hv_s)
            # hack reward begin
            rewards = torch.stack(rewards)
            #~ if norm_hv:
                #~ hv_c = (hv_c - hv_c.mean())/(hv_c.std()+1e-8)
            #~ else:
                #~ hv_c *= scale_hv
            #~ rewards.view(-1, 1)[hv_reward_idx] += torch.FloatTensor(hv_c).unsqueeze(1).to(device)
            ndf_ = (torch.FloatTensor(hv_c)>0)
            hv_ = ndf_.float() * hv_s * hv_scale    # 1e-2
            rewards.view(-1, 1)[hv_reward_idx] += hv_.unsqueeze(1).to(device)
            # hack reward end
            if train_valid and valid_mode:
                writer.add_scalar(tsbx_id+'_mo/hv_valid', hv_s, j-1)
                writer.add_scalars(tsbx_id+'_mo/ihv_valid', {'mean':np.mean(hv_c), 
                                        'min':np.min(hv_c), 'max':np.max(hv_c)}, j-1)
            else:
                writer.add_scalar(tsbx_id+'_mo/hv', hv_s, j)
                writer.add_scalars(tsbx_id+'_mo/ihv', {'mean':np.mean(hv_c), 
                                        'min':np.min(hv_c), 'max':np.max(hv_c)}, j)
            if len(moo_pop)>1:
                t_plot = time.time()
                if train_valid and valid_mode:
                    writer.add_figure(tsbx_id+"valid_pareto", render_moo(moo_pop), j-1)
                    writer.add_figure(tsbx_id+"valid_pareto_gail", render_moo(gail_moo_pop), j-1)
                else:
                    writer.add_figure(tsbx_id+"pareto", render_moo(moo_pop), j)
                    writer.add_figure(tsbx_id+"pareto_gail", render_moo(gail_moo_pop), j)
                t_plot = time.time() - t_plot
                print(f"plot time: {t_plot*1000:.0f} ms")
                
        # hv end
        
        # hv begin
        #~ rewards = torch.stack(rewards)
        if len(moo_pop)>1 and ((moo_mode and ref_p is not None and not use_gail) or (not moo_mode and valid_mode)):
            hv = pg.hypervolume(moo_pop)
            #~ ref_p = [1, 51, 1]              # DeepseaEnergy
            #~ ref_p = [1, 51]              # DeapSea
            #~ ref_p = [41, 1001]              # Moo
            #~ ref_p = hv.refpoint(offset=0.1)
            hv_c = hv.contributions(ref_p)
            hv_s = hv.compute(ref_p)
            # hack reward begin
            rewards = torch.stack(rewards)
            #~ if norm_hv:
                #~ hv_c = (hv_c - hv_c.mean())/(hv_c.std()+1e-8)
            #~ else:
                #~ hv_c *= scale_hv
            #~ hv_c *= 0.01    # DeapSea
            #~ hv_c *= 0.001    # DeepseaEnergy
            #~ rewards.view(-1, 1)[hv_reward_idx] = torch.FloatTensor(hv_c).unsqueeze(1).to(device)
            #~ rewards.view(-1, 1)[hv_reward_idx] += torch.FloatTensor(hv_c).unsqueeze(1).to(device)
            #~ ndf_ = (torch.FloatTensor(hv_c)>0).float().unsqueeze(1).to(device)
            ndf_ = (torch.FloatTensor(hv_c)>0)
            #~ zeros_ = (~ndf_).float() * -1
            #~ zeros_ = (~ndf_).float() * -1.5         # 1.2 1.3 1.5
            #~ hv_ = zeros_ + ndf_.float() * hv_s * 1e-3
            hv_ = ndf_.float() * hv_s * 1e-3
            #~ print(rewards.view(-1, 1)[hv_reward_idx].shape, hv_.unsqueeze(1).shape, len(hv_reward_idx), len(moo_pop))
            if moo_mode:
                rewards.view(-1, 1)[hv_reward_idx] += hv_.unsqueeze(1).to(device)
            #~ rewards = rewards.unbind()
            # hack reward end
            if not valid_mode:
                writer.add_scalar(tsbx_id+'_mo/hv', hv_s, j)
                #~ writer.add_scalars(tsbx_id+'_mo/ihv', {'mean':np.mean(hv_c), 
                                        #~ 'min':np.min(hv_c), 'max':np.max(hv_c)}, j)
            else:
                writer.add_scalar(tsbx_id+'_mo/hv_valid', hv_s, j)
                #~ writer.add_scalars(tsbx_id+'_mo/ihv_valid', {'mean':np.mean(hv_c), 
                                        #~ 'min':np.min(hv_c), 'max':np.max(hv_c)}, j)
            
            if len(moo_pop)>1:
                t_plot = time.time()
                pareto = 'pareto_valid' if valid_mode else 'pareto'
                writer.add_figure(tsbx_id+pareto, render_moo(moo_pop, hv_s), j)
                t_plot = time.time() - t_plot
                print(f"plot time: {t_plot*1000:.0f} ms")
        # hv end
        
        # reward+hv begin
        if moo_mode:
            if train_valid and valid_mode:
                ep_rws_hv = rewards.view(-1, 1)[ep_reward_idx].numpy()
                writer.add_scalars(tsbx_id+'_mo/reward_hv_valid', {'mean':np.mean(ep_rws_hv), 
                                                'min':np.min(ep_rws_hv), 'max':np.max(ep_rws_hv)}, j-1)
            else:
                ep_rws_hv = rewards.view(-1, 1)[ep_reward_idx].numpy()
                writer.add_scalars(tsbx_id+'_mo/reward_hv_train', {'mean':np.mean(ep_rws_hv), 
                                                'min':np.min(ep_rws_hv), 'max':np.max(ep_rws_hv)}, j)
        # reward+hv end
        
        if rand_orth_reg and not loss_reg:
            lossReg = model.actor.random_proj_reg(num_envs, perm=1)
            loss_reg.append(lossReg)
        
        # dump traj begin
        if valid_mode and hasattr(rollout_dataset, 'save'):
            rollout_dataset.save(start, j-1, np.mean(episode_rewards), np.max(episode_rewards))
        # dump traj end
        
        if train_valid and valid_mode:
            writer.add_scalars(tsbx_id+'_train/reward_valid', {'mean':np.mean(episode_rewards), 
                                    'min':np.min(episode_rewards), 'max':np.max(episode_rewards)}, j)
            writer.add_scalar(tsbx_id+'_train/reward_env0_valid', env0_rws, j)
            continue
        # pure valid mode
        elif valid_mode:
            writer.add_scalars(tsbx_id+'_train/reward', {'mean':np.mean(episode_rewards), 
                                    'min':np.min(episode_rewards), 'max':np.max(episode_rewards)}, j)
            writer.add_scalar(tsbx_id+'_train/reward_env0', env0_rws, j)
            # dump traj begin
            if hasattr(rollout_dataset, 'save'):
                rollout_dataset.save(start, j-1, np.mean(episode_rewards), np.max(episode_rewards))
            # dump traj end
            if moo_mode:
                np.savez(os.path.join("trained_z", time.strftime(f'valid_%Y%m%d_%H%M%S_{j+1}', time.localtime()) + ".npz"), 
                            y=moo_pop, z=moo_z, x=moo_x)
            for _ in range(len(best_z)-100):
                best_z.popitem(0)
            if j==valid_ep-1 and valid_z is None:
                zs = []
                for jz in range(min(10, len(best_z))):
                    zz = best_z.popitem()
                    zs.append(zz[1])
                    writer.add_text(tsbx_id+'best_z', 'r: %.1f, z: %s'%zz, jz)
                torch.save({"zs": zs}, 
                            os.path.join("trained_z", time.strftime('%Y%m%d_%H%M%S', time.localtime()) + ".pt"))
                print('bye')
                break
            continue
        
        t_optm = time.time()
        # V Begin
        if use_critic:
            next_state = torch.FloatTensor(next_state).to(device)
            with torch.no_grad():
                next_value = model.value(next_state)
        else:
            next_value = 0
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        entropys = torch.cat(entropys)
        entropy = entropys.mean()
        
        optimizer.zero_grad()
        
        if use_critic:
            values    = torch.cat(values)
            advantage = returns - values
            actor_loss  = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            loss = actor_loss + coef_value * critic_loss
        else:
            advantage = returns
            actor_loss  = -(log_probs * advantage.detach()).mean()
            critic_loss = torch.tensor(0)
            loss = actor_loss
        
        if rand_orth_reg:
            loss_orth = torch.stack(loss_reg).mean()
        if rand_orth_reg==1:
            loss +=  loss_orth * coeff_orth
        
        if use_gail and policy_grad_pen:
            loss += discr.policy_grad_pen(gail_train_loader, rollouts, model, num_inputs, output_size)
        
        loss.backward()
        # check grads begin
        if j%25==0 and log_grads:
            for name, param in model.named_parameters():
                name = name.split('.')
                #~ name = '.'.join(name[:-1]) + '/' + name[-1]
                name = '.'.join(name[:1]) + '/' + '.'.join(name[1:])
                writer.add_histogram(tsbx_id+'model.%s'%name, param.data.detach().cpu().numpy(), j)
                writer.add_histogram(tsbx_id+'model.%s.grads'%name, param.grad.detach().cpu().numpy(), j)
        if use_kfac:
            preconditioner.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        # check grads end
        optimizer.step()
        # debug begin
        t_optm = time.time() - t_optm
        print(f"optm time: {t_optm*1000:.0f} ms")
        t_log = time.time()
        #~ if hasattr(model, 'reset'):
            #~ model.reset()
        # debug end
        
        # layer_rotation begin
        if use_layer_rotation in (1,3):
            model.set_layer_rotation_lr(layer_rotation_groups, lr)
        # layer_rotation end
        # schedule lr begin
        if use_scheduler_lr:
            if use_scheduler_lr==1:
                update_linear_schedule(optimizer, j, num_updates, lr)
            elif use_scheduler_lr==2:
                scheduler_lr.step()
            if j>=num_scheduler_lr:
                set_lr(optimizer, lr)
                use_scheduler_lr = 0
        if use_entropy_decay:
            coef_entropy = update_entropy_schedule(j, use_entropy_decay, initial_entropy, min_entropy)
        # schedule lr end
        
        total_num_steps = (j + 1) * num_envs * num_steps
        
        #~ if j%save_interval==0 and j>0:
        if (j<100 and j%2==0) \
                or ((100<=j<=600 or 1000<=j<=1200 or 1500<=j<=1700 or 2000<=j<=2200) and j%5==0) \
                or j%save_interval==0:
            ob_rms = getattr(get_vec_normalize(envs), 'ob_rms', None)
            model.save(start, j, np.mean(episode_rewards), np.max(episode_rewards), ob_rms, optimizer.state_dict())
            if use_gail and use_gail!=6:
                discr.save(start, j, np.mean(episode_rewards), np.max(episode_rewards))
        
        if j % log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, frames {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}\n".
                format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        np.mean(episode_rewards),
                        np.median(episode_rewards),
                        np.min(episode_rewards),
                        np.max(episode_rewards),
                        entropy.item(),
                        critic_loss.item(), actor_loss.item()))
        writer.add_scalars(tsbx_id+'_train/reward', {'mean':np.mean(episode_rewards), 
                                'min':np.min(episode_rewards), 'max':np.max(episode_rewards)}, j)
        
        #~ if log_debug:
            #~ writer.add_scalar(tsbx_id+'model/gloss', gloss.item(), total_num_steps)
            #~ writer.add_scalar(tsbx_id+'debug/log_probs', log_probs.detach().mean(), total_num_steps)
            #~ writer.add_scalar(tsbx_id+'debug/advantage', advantage.detach().mean(), total_num_steps)
            #~ writer.add_scalar(tsbx_id+'debug/values', values.detach().mean(), total_num_steps)
            #~ writer.add_scalar(tsbx_id+'debug/returns', returns.mean(), total_num_steps)
        writer.add_scalar(tsbx_id+'model/actor_loss', actor_loss.item(), j)
        writer.add_scalar(tsbx_id+'model/critic_loss', critic_loss.item(), j)
        writer.add_scalar(tsbx_id+'_train/entropy', entropy.item(), j)
        if rand_orth_reg:
            writer.add_scalar(tsbx_id+'model/rand_orth_reg', loss_orth.item(), j)
        if use_entropy_decay:
            writer.add_scalar(tsbx_id+'_train/coef_entropy', coef_entropy, j)
        if use_scheduler_lr:
            writer.add_scalars(tsbx_id+'_train/lr', get_lr(optimizer), j)
        if use_layer_rotation in (1,3):
            writer.add_scalars(tsbx_id+'_train/lr', get_lr_layer_rotation(optimizer), j)
        if j%10==0 and hasattr(model, 'get_layer_rotation'):
            writer.add_scalars(tsbx_id+'_train/layer_rotation', model.get_layer_rotation(), j)
        #~ if j%100==0:
            #~ writer.add_histogram(tsbx_id+'generated model', model.vis(), j)
            #~ writer.add_image(tsbx_id+'Generated model', model.vis(), j)
            #~ writer.add_image(tsbx_id+'Generated model', make_grid(model.vis()[:20], nrow=4, normalize=True), j)
        t_log = time.time() - t_log
        #~ print(f"t_log time: {t_log*1000:.0f} ms")
        if nan_check:
            break


if __name__=='__main__':
    main()

