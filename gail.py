import os, time
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch import autograd

from baselines.common.running_mean_std import RunningMeanStd
from mmd_loss import MMD_loss


class RND_Critic(nn.Module):
    def __init__(self, ob_size, ac_size, device, lr=5e-4, 
                 rnd_hid_size=100, rnd_hid_layer=4, hid_size=100, hid_layer=1,
                 out_size=100, scale=2500.0, offset=0., reward_scale=1.0):
        super().__init__()
        
        self.scale = scale
        self.offset = offset
        self.out_size = out_size
        self.rnd_hid_size = rnd_hid_size
        self.rnd_hid_layer = rnd_hid_layer
        self.hid_size = hid_size
        self.hid_layer = hid_layer
        self.reward_scale = reward_scale
        
        self.device = device
        
        self.state_only = False
        input_dim = obs_dim if self.state_only else ob_size + ac_size
        
        act_fn = nn.Tanh
        #~ act_fn = nn.ReLU
        
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hid_size), act_fn(),
            nn.Linear(hid_size, hid_size), act_fn(),
            nn.Linear(hid_size, out_size)).to(device)
        
        self.rnd_feat = nn.Sequential(
            nn.Linear(input_dim, rnd_hid_size), act_fn(),
            nn.Linear(rnd_hid_size, rnd_hid_size), act_fn(),
            nn.Linear(rnd_hid_size, rnd_hid_size), act_fn(),
            nn.Linear(rnd_hid_size, rnd_hid_size), act_fn(),
            nn.Linear(rnd_hid_size, out_size)).to(device)
        
        self.optimizer = torch.optim.Adam(self.feat.parameters(), lr=lr)
    
    def save(self, save_path='rnd_model.pt'):
        torch.save({"model":self.state_dict()}, save_path)

    def load(self, path='rnd_model.pt'):
        data = torch.load(path)
        self.load_state_dict(data["model"])
    
    def update(self, expert_loader):
        self.train()

        loss = 0
        n = 0
        
        for expert_batch in expert_loader:
            expert_state, expert_action, _ = expert_batch
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            
            feat = self.feat(torch.cat([expert_state, expert_action], dim=1))
            rnd_feat = self.rnd_feat(torch.cat([expert_state, expert_action], dim=1))
            
            feat_loss = torch.square(feat-rnd_feat).mean()
            
            loss += feat_loss.item()
            n += 1

            self.optimizer.zero_grad()
            feat_loss.backward()
            self.optimizer.step()
        
        return {'loss': loss/n}
    
    def get_raw_reward(self, ob, ac):
        with torch.no_grad():
            self.eval()
            feat = self.feat(torch.cat([ob, ac], dim=1))
            rnd_feat = self.rnd_feat(torch.cat([ob, ac], dim=1))
            reward = torch.mean(torch.square(feat - rnd_feat), dim=1, keepdims=True) * self.scale
            return reward
    
    def get_reward(self, ob, ac):
        with torch.no_grad():
            self.eval()
            feat = self.feat(torch.cat([ob, ac], dim=1))
            rnd_feat = self.rnd_feat(torch.cat([ob, ac], dim=1))
            reward = self.reward_scale*torch.exp(self.offset - torch.mean(torch.square(feat - rnd_feat), dim=1, keepdims=True) * self.scale)
            return reward


class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, device, lr=1e-3, discount=0.99, input_emb=None):
        super().__init__()
        
        self.device = device
        self.gamma = discount
        
        self.use_value = False
        #~ self.use_value = True
        
        self.state_only = False
        #~ self.state_only = True
        
        input_dim = obs_dim if self.state_only else obs_dim + act_dim
        
        if input_emb: input_dim = input_emb
        
        act_fn = nn.Tanh
        #~ act_fn = nn.ReLU
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            #~ nn.BatchNorm1d(hidden_dim), 
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim), act_fn(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()
        
        if self.use_value:
            self.value_fn = nn.Sequential(
                nn.Linear(obs_dim, 32), act_fn(),
                nn.Linear(32, 32), act_fn(),
                nn.Linear(32, 1)).to(device)

            self.value_fn.train()
        
        self.triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
        #~ self.triplet_loss = nn.TripletMarginWithDistanceLoss(margin=0.3, distance_function=nn.CrossEntropyLoss())
        
        self.mmd_loss = MMD_loss()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #~ self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=lr)
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
        
        print(self)
    
    def __str__(self):
        return f'D: {self.obs_dim}x32 32x32 32x1'
    
    def save(self, t_start, j, reward, reward1):
        save_path = os.path.join("trained_models", time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start)))
        os.makedirs(save_path, exist_ok=True)
        model_name = f"EP_{j}_{reward:.0f}_{reward1:.0f}_d"
        torch.save({"model":self.state_dict(), "opt_state":self.optimizer.state_dict(), "ret_rms":self.ret_rms}, 
                            os.path.join(save_path, model_name + ".pt"))

    def load(self, path):
        data = torch.load(path)
        self.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt_state"])
        self.ret_rms = data["ret_rms"]
    
    def d_score(self, d):
        with torch.no_grad():
            s = torch.sigmoid(d)
            #~ score = s.log() - (1 - s).log()
            score = s
            return score.mean().item()
    
    def cosine_similarity(self, input1, input2):
        return F.cosine_similarity(input1, input2).mean().item()
    
    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        
        if self.state_only:
            expert_data = expert_state
            policy_data = policy_state
        else:
            expert_data = torch.cat([expert_state, expert_action], dim=1)
            policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    def compute_grad_pen_rnn(self,
                         expert_state,
                         policy_state,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        
        expert_data = expert_state
        policy_data = policy_state

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        #~ mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    def compute_grad_pen_policy(self,
                            policy,
                            expert_state,
                            expert_action,
                            policy_state,
                            policy_action,
                            lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True
        
        mixup_state = mixup_data[:, :expert_state.size(1)]
        mixup_action = mixup_data[:, expert_state.size(1):]
        
        disc = policy.eval_grad(mixup_state, mixup_action)
        
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
    
    def policy_grad_pen(self, expert_loader, rollouts, policy=None, obs_dim=3, act_dim=3):
        loss = 0
        n = 0
        
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_state, policy_action, _ = policy_batch
            
            expert_state, expert_action, _ = expert_batch
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            
            policy_state = policy_state.reshape(-1, obs_dim)
            policy_action = policy_action.reshape(-1, act_dim)
            
            expert_state = expert_state.reshape(-1, obs_dim)
            expert_action = expert_action.reshape(-1, act_dim)
            
            grad_pen = self.compute_grad_pen_policy(policy, expert_state, expert_action,
                                             policy_state, policy_action)

            loss += grad_pen
            n += 1

        return loss/n
    
    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        loss = 0
        n = 0
        
        d_expert = 0
        d_policy = 0
        
        d_cosine = 0
        d_mmd = 0
        
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_state, policy_action, _ = policy_batch
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))
            
            expert_state, expert_action, _ = expert_batch
            if obsfilt:
                expert_state = obsfilt(expert_state.numpy()[:,:], update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            
            d_expert += self.d_score(expert_d)
            d_policy += self.d_score(policy_d)
            
            d_cosine += self.cosine_similarity(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            d_mmd += self.mmd_loss(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            
        return {'loss': loss/n, 'd_expert': d_expert/n, 'd_policy': d_policy/n, 'd_cosine': d_cosine/n, 'd_mmd': d_mmd/n}
    
    def update_triplet(self, expert_loader, rollouts, rollouts1):
        self.train()

        loss = 0
        n = 0
        d_expert = 0
        d_policy = 0
        
        cx = nn.KLDivLoss(reduction='batchmean')
        #~ cx = F.binary_cross_entropy_with_logits
        def triplet_loss(a, p, n, margin=0.3):
            a = torch.sigmoid(a).view(1, -1)
            p = torch.sigmoid(p).view(1, -1)
            n = torch.sigmoid(n).view(1, -1)
            return torch.clamp(cx(a.log(),p) - cx(a.log(),n) + margin, min=0)
        
        for expert_batch, policy_batch, policy_batch1 in zip(expert_loader, rollouts, rollouts1):
            policy_state, policy_action, _ = policy_batch
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))
            
            policy_state1, policy_action1, _ = policy_batch1
            policy_d1 = self.trunk(
                torch.cat([policy_state1, policy_action1], dim=1))
            
            expert_state, expert_action, _ = expert_batch
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))
            
            #~ negative = torch.randperm(policy_d.size(0))
            #~ gail_loss = self.triplet_loss(policy_d, expert_d, policy_d[negative])
            #~ gail_loss = self.triplet_loss(policy_d, expert_d, policy_d1)
            gail_loss = triplet_loss(policy_d, expert_d, policy_d1)
            
            #~ expert_loss = F.binary_cross_entropy_with_logits(
                #~ expert_d,
                #~ torch.ones(expert_d.size()).to(self.device))
            #~ policy_loss = F.binary_cross_entropy_with_logits(
                #~ policy_d,
                #~ torch.zeros(policy_d.size()).to(self.device))

            #~ gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            
            d_expert += self.d_score(expert_d)
            d_policy += self.d_score(policy_d)
            
        return {'loss': loss/n, 'd_expert': d_expert/n, 'd_policy': d_policy/n}
    
    def update_energy(self, expert_loader, rollouts, policy=None, value_fn=None, obsfilt=None):
        self.train()
        
        _log_p_policy = 0
        _log_p_expert = 0
        _log_q_policy = 0
        _log_q_expert = 0
        _d_policy = 0
        _d_expert = 0
        
        d_cosine = 0
        d_mmd = 0
        
        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_state, policy_action, policy_obs_next = policy_batch
            policy_lprobs = policy.eval(policy_state, policy_action)
            
            if self.state_only:
                policy_d = self.trunk(policy_state)
            else:
                policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))
            
            expert_state, expert_action, expert_obs_next = expert_batch
            expert_lprobs = policy.eval(expert_state, expert_action)
            
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            expert_obs_next = expert_obs_next.to(self.device)
            
            if self.state_only:
                expert_d = self.trunk(expert_state)
            else:
                expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))
            
            if self.use_value:
                expert_v = self.value_fn(expert_state)
                expert_vn = self.value_fn(expert_obs_next)
                policy_v = self.value_fn(policy_state)
                policy_vn = self.value_fn(policy_obs_next)
                
                #~ expert_v = value_fn(expert_state)
                #~ expert_vn = value_fn(expert_obs_next)
                #~ policy_v = value_fn(policy_state)
                #~ policy_vn = value_fn(policy_obs_next)
                
                log_p_policy = policy_d + self.gamma*policy_vn - policy_v
                log_p_expert = expert_d + self.gamma*expert_vn - expert_v
            else:
                log_p_policy = policy_d
                log_p_expert = expert_d
            
            #~ log_p_policy = -policy_d
            #~ log_p_expert = -expert_d
            
            log_q_policy = policy_lprobs
            log_q_expert = expert_lprobs
            #~ log_q_policy = policy_lprobs.sum(dim=1, keepdim=True)       # failed
            #~ log_q_expert = expert_lprobs.sum(dim=1, keepdim=True)       
            
            #~ log_pq_policy = torch.logaddexp(log_p_policy, log_q_policy)
            #~ log_pq_expert = torch.logaddexp(log_p_expert, log_q_expert)
            
            log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
            log_pq_expert = torch.log(log_p_expert.exp() + log_q_expert.exp())
            
            d_policy = torch.exp(log_p_policy-log_pq_policy).mean()
            d_expert = torch.exp(log_p_expert-log_pq_expert).mean()
            
            #~ log_p_tau = torch.cat([log_p_policy, log_p_expert])
            #~ log_q_tau = torch.cat([log_q_policy, log_q_expert])
            #~ log_pq = torch.cat([log_pq_policy, log_pq_expert])
            #~ d_tau = torch.exp(log_p_tau-log_pq)
            
            expert_loss = log_p_expert - log_pq_expert
            policy_loss = log_q_policy - log_pq_policy
            
            gail_loss = - expert_loss.mean() - policy_loss.mean()
            
            #~ batch_size = policy_state.size(0)
            #~ labels = torch.zeros(batch_size*2, 1)
            #~ labels[batch_size:] = 1
            
            #~ gail_loss = -torch.mean(labels*(log_p_tau-log_pq) + (1-labels)*(log_q_tau-log_pq))
            
            #~ expert_loss = F.binary_cross_entropy_with_logits(
                #~ expert_d,
                #~ torch.ones(expert_d.size()).to(self.device))
            #~ policy_loss = F.binary_cross_entropy_with_logits(
                #~ policy_d,
                #~ torch.zeros(policy_d.size()).to(self.device))

            #~ gail_loss = expert_loss + policy_loss
            
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            #~ loss += (gail_loss + grad_pen).item()
            loss += gail_loss.item()
            n += 1
            
            _log_p_policy += log_p_policy.mean().item()
            _log_p_expert += log_p_expert.mean().item()
            _log_q_policy += log_q_policy.mean().item()
            _log_q_expert += log_q_expert.mean().item()
            _d_policy += d_policy.item()
            _d_expert += d_expert.item()
            
            self.optimizer.zero_grad()
            (gail_loss+grad_pen).backward()
            #~ gail_loss.backward()
            self.optimizer.step()
            
            d_cosine += self.cosine_similarity(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            d_mmd += self.mmd_loss(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            
        return {'loss': loss/n, 'log_p_policy': _log_p_policy/n, 'log_p_expert': _log_p_expert/n, 
                    'log_q_policy': _log_q_policy/n, 'log_q_expert': _log_q_expert/n, 
                    'd_policy': _d_policy/n, 'd_expert': _d_expert/n, 'd_cosine': d_cosine/n, 'd_mmd': d_mmd/n}
    
    def predict_energy(self, state, action, log_p, obs_next, value_fn, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            if self.state_only:
                energy = self.trunk(state)
            else:
                energy = self.trunk(torch.cat([state, action], dim=1))
            
            if 0:
            #~ if 1:
                #~ policy_v = self.value_fn(state)
                #~ policy_vn = self.value_fn(obs_next)
                policy_v = value_fn(state)
                policy_vn = value_fn(obs_next)
                
                log_p_policy = energy + self.gamma*policy_vn - policy_v
                
                #~ log_p_policy = energy
                
                log_q_policy = log_p
                log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
                d_policy = torch.exp(log_p_policy-log_pq_policy)
                
                #~ reward = d_policy.mean(dim=1, keepdim=True)
                
                score = torch.log(d_policy) - torch.log(1-d_policy)
                #~ reward = score.sum(dim=1, keepdim=True)
                reward = score.mean(dim=1, keepdim=True)
            else:
                #~ reward = (energy - log_p).mean(dim=1, keepdim=True)
                reward = energy
            
            #~ return reward
            
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_energy_r(self, state, action, log_p, obs_next, value_fn, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            if self.state_only:
                energy = self.trunk(state)
            else:
                energy = self.trunk(torch.cat([state, action], dim=1))
            
            if 0:
            #~ if 1:
                policy_v = self.value_fn(state)
                policy_vn = self.value_fn(obs_next)
                #~ policy_v = value_fn(state)
                #~ policy_vn = value_fn(obs_next)
                
                log_p_policy = energy + self.gamma*policy_vn - policy_v
                
                #~ log_p_policy = energy
                
                log_q_policy = log_p
                log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
                d_policy = torch.exp(log_p_policy-log_pq_policy)
                
                #~ reward = d_policy.mean(dim=1, keepdim=True)
                
                score = torch.log(d_policy) - torch.log(1-d_policy)
                reward = score.sum(dim=1, keepdim=True)
                #~ reward = score.mean(dim=1, keepdim=True)
            else:
                #~ reward = (energy - log_p).mean(dim=1, keepdim=True)
                reward = energy
            
            #~ return reward
            
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_reward_raw(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            
            return reward.clamp(-10, 10)
    
    def update_energy_traj(self, expert_loader, rollouts, policy=None, value_fn=None, obsfilt=None):
        self.train()
        
        _log_p_policy = 0
        _log_p_expert = 0
        _log_q_policy = 0
        _log_q_expert = 0
        _d_policy = 0
        _d_expert = 0
        
        d_cosine = 0
        d_mmd = 0
        
        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            #~ policy_state, policy_action, policy_mask = policy_batch
            #~ policy_state = policy_state.view(policy_state.size(0), -1, self.obs_dim)
            #~ policy_action = policy_action.view(policy_action.size(0), -1, self.act_dim)
            #~ policy_lprobs = policy.eval(policy_state.view(-1, self.obs_dim), 
                                        #~ policy_action.view(-1, self.act_dim)).view(policy_action.size(0), -1, self.act_dim)
            
            #~ policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=2))
            
            #~ expert_state, expert_action, _ = expert_batch
            #~ expert_state = expert_state.view(expert_state.size(0), -1, self.obs_dim)
            #~ expert_action = expert_action.view(expert_action.size(0), -1, self.act_dim)
            #~ expert_lprobs = policy.eval(expert_state.view(-1, self.obs_dim), 
                                        #~ expert_action.view(-1, self.act_dim)).view(expert_action.size(0), -1, self.act_dim)
            
            #~ expert_state = expert_state.to(self.device)
            #~ expert_action = expert_action.to(self.device)
            
            #~ expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=2))
            
            policy_state, policy_action, policy_mask = policy_batch
            bs_policy = policy_state.size(0)
            policy_state = policy_state.view(-1, self.obs_dim)
            policy_action = policy_action.view(-1, self.act_dim)
            
            expert_state, expert_action, _ = expert_batch
            bs_expert = expert_state.size(0)
            expert_state = expert_state.view(-1, self.obs_dim)
            expert_action = expert_action.view(-1, self.act_dim)
            
            #~ limit_std = 0.5
            limit_std = 0.3
            #~ limit_std = 0.2
            #~ limit_std = 0.15
            # (batch, steps, act_dim)
            expert_lprobs = policy.eval(expert_state, expert_action, limit_std).view(bs_expert, -1, self.act_dim)
            policy_lprobs = policy.eval(policy_state, policy_action, limit_std).view(bs_policy, -1, self.act_dim)
            
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            
            # (batch, steps, 1)
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1)).view(bs_policy, -1, 1)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1)).view(bs_expert, -1, 1)
            
            # (batch, steps, 1)
            policy_d = torch.where(policy_mask, policy_d, torch.zeros_like(policy_d))
            policy_lprobs = torch.where(policy_mask.expand_as(policy_lprobs), policy_lprobs, torch.zeros_like(policy_lprobs))
            
            # (batch, 1)
            log_p_policy = torch.sum(-policy_d, dim=1)
            log_p_expert = torch.sum(-expert_d, dim=1)
            
            # (batch, 1)
            log_q_policy = torch.sum(policy_lprobs, dim=1)
            log_q_expert = torch.sum(expert_lprobs, dim=1)
            #~ log_q_policy = torch.sum(policy_lprobs, dim=1, keepdim=True)
            #~ log_q_expert = torch.sum(expert_lprobs, dim=1, keepdim=True)
            
            log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
            log_pq_expert = torch.log(log_p_expert.exp() + log_q_expert.exp())
            
            d_policy = torch.exp(log_p_policy-log_pq_policy).mean()
            d_expert = torch.exp(log_p_expert-log_pq_expert).mean()
            
            #~ log_p_tau = torch.cat([log_p_policy, log_p_expert])
            #~ log_q_tau = torch.cat([log_q_policy, log_q_expert])
            #~ log_pq = torch.cat([log_pq_policy, log_pq_expert])
            #~ d_tau = torch.exp(log_p_tau-log_pq)
            
            expert_loss = log_p_expert - log_pq_expert
            policy_loss = log_q_policy - log_pq_policy
            
            gail_loss = - expert_loss.mean() - policy_loss.mean()
            
            grad_pen = self.compute_grad_pen(expert_state.view(-1, self.obs_dim), 
                                            expert_action.view(-1, self.act_dim),
                                            policy_state.view(-1, self.obs_dim), 
                                            policy_action.view(-1, self.act_dim))

            #~ loss += (gail_loss + grad_pen).item()
            loss += gail_loss.item()
            n += 1
            
            _log_p_policy += log_p_policy.mean().item()
            _log_p_expert += log_p_expert.mean().item()
            _log_q_policy += log_q_policy.mean().item()
            _log_q_expert += log_q_expert.mean().item()
            _d_policy += d_policy.item()
            _d_expert += d_expert.item()
            
            # debug begin
            check_nan = False
            #~ check_nan = True
            nan_check = 0
            def check_param_nan():
                param_nan = 0
                for name, param in self.named_parameters():
                    name = name.split('.')
                    name = '.'.join(name[:1]) + '/' + '.'.join(name[1:])
                    if torch.isnan(param.data.detach().cpu()).any():
                        print(f'{name} is NaN !!!')
                        param_nan += 1
                    if torch.isnan(param.grad.detach().cpu()).any():
                        #~ torch.nan_to_num_(param.grad)
                        print(f'{name}.grad is NaN !!!')
                        param_nan += 1
                return param_nan
            def check_nan(x, name='x', title='update_traj', show=None):
                if torch.isnan(x).any():
                    print(f"{title}: {name} is NaN !!!")
                    if show is not None:
                        print(show)
                    check_param_nan()
                    return 1
                return 0
            if check_nan:
                nan_check += check_nan(policy_d, 'policy_d', show=None)
                nan_check += check_nan(expert_d, 'expert_d', show=None)
                nan_check += check_nan(log_p_policy, 'log_p_policy')
                nan_check += check_nan(log_p_expert, 'log_p_expert')
                nan_check += check_nan(log_q_policy, 'log_q_policy')
                nan_check += check_nan(log_q_expert, 'log_q_expert')
                nan_check += check_nan(gail_loss, 'gail_loss')
                nan_check += check_nan(grad_pen, 'grad_pen')
            if nan_check:
                return
            # debug end
            
            self.optimizer.zero_grad()
            (gail_loss+grad_pen).backward()
            #~ gail_loss.backward()
            #~ nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            #~ self.optimizer.step()
            # debug begin
            #~ print('check_param_nan before backward')
            if check_nan and check_param_nan():
                #~ print(torch.cat([policy_action, expert_action], dim=1))
                print(policy_action)
                return
            #~ nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            # debug end
            
            d_cosine += self.cosine_similarity(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            d_mmd += self.mmd_loss(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            
        return {'loss': loss/n, 'log_p_policy': _log_p_policy/n, 'log_p_expert': _log_p_expert/n, 
                    'log_q_policy': _log_q_policy/n, 'log_q_expert': _log_q_expert/n, 
                    'd_policy': _d_policy/n, 'd_expert': _d_expert/n, 'd_cosine': d_cosine/n, 'd_mmd': d_mmd/n}
    
    def update_energy_traj7(self, expert_loader, rollouts, policy=None, value_fn=None, obsfilt=None):
        self.train()
        
        _log_p_policy = 0
        _log_p_expert = 0
        _log_q_policy = 0
        _log_q_expert = 0
        _d_policy = 0
        _d_expert = 0
        
        d_cosine = 0
        d_mmd = 0
        
        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            
            policy_state, policy_action, policy_mask = policy_batch
            bs_policy = policy_state.size(0)
            policy_state_step = policy_state.view(-1, self.obs_dim)
            policy_action_step = policy_action.view(-1, self.act_dim)
            
            expert_state, expert_action, _ = expert_batch
            bs_expert = expert_state.size(0)
            expert_state_step = expert_state.view(-1, self.obs_dim)
            expert_action_step = expert_action.view(-1, self.act_dim)
            
            #~ limit_pxpert = 0.5      # work
            limit_pxpert = 0.3      # work
            #~ limit_pxpert = 0.2      # test
            limit_policy = 0.3      # work
            #~ limit_policy = 0.15      # test
            limit_policy = limit_pxpert = 0.2
            # (batch, steps, act_dim)
            expert_lprobs = policy.eval(expert_state_step, expert_action_step, limit_pxpert).view(bs_expert, -1, self.act_dim)
            policy_lprobs = policy.eval(policy_state_step, policy_action_step, limit_policy).view(bs_policy, -1, self.act_dim)
            
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            
            # (batch, 1)
            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))
            
            # (batch, steps, 1)
            #~ policy_d = torch.where(policy_mask, policy_d, torch.zeros_like(policy_d))
            policy_lprobs = torch.where(policy_mask.expand_as(policy_lprobs), policy_lprobs, torch.zeros_like(policy_lprobs))
            
            # (batch, 1)
            log_p_policy = -policy_d
            log_p_expert = -expert_d
            #~ log_p_policy = torch.sum(-policy_d, dim=1)
            #~ log_p_expert = torch.sum(-expert_d, dim=1)
            
            # (batch, 1)
            #~ log_q_policy = torch.mean(policy_lprobs, dim=1)
            #~ log_q_expert = torch.mean(expert_lprobs, dim=1)
            log_q_policy = torch.sum(policy_lprobs, dim=1)
            log_q_expert = torch.sum(expert_lprobs, dim=1)
            #~ log_q_policy = torch.sum(policy_lprobs, dim=1, keepdim=True)
            #~ log_q_expert = torch.sum(expert_lprobs, dim=1, keepdim=True)
            
            log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
            log_pq_expert = torch.log(log_p_expert.exp() + log_q_expert.exp())
            
            d_policy = torch.exp(log_p_policy-log_pq_policy).mean()
            d_expert = torch.exp(log_p_expert-log_pq_expert).mean()
            
            #~ log_p_tau = torch.cat([log_p_policy, log_p_expert])
            #~ log_q_tau = torch.cat([log_q_policy, log_q_expert])
            #~ log_pq = torch.cat([log_pq_policy, log_pq_expert])
            #~ d_tau = torch.exp(log_p_tau-log_pq)
            
            expert_loss = log_p_expert - log_pq_expert
            policy_loss = log_q_policy - log_pq_policy
            
            gail_loss = - expert_loss.mean() - policy_loss.mean()
            
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                            policy_state, policy_action)

            #~ loss += (gail_loss + grad_pen).item()
            loss += gail_loss.item()
            n += 1
            
            _log_p_policy += log_p_policy.mean().item()
            _log_p_expert += log_p_expert.mean().item()
            _log_q_policy += log_q_policy.mean().item()
            _log_q_expert += log_q_expert.mean().item()
            _d_policy += d_policy.item()
            _d_expert += d_expert.item()
            
            # debug begin
            check_nan = False
            #~ check_nan = True
            nan_check = 0
            def check_param_nan():
                param_nan = 0
                for name, param in self.named_parameters():
                    name = name.split('.')
                    name = '.'.join(name[:1]) + '/' + '.'.join(name[1:])
                    if torch.isnan(param.data.detach().cpu()).any():
                        print(f'{name} is NaN !!!')
                        param_nan += 1
                    if torch.isnan(param.grad.detach().cpu()).any():
                        #~ torch.nan_to_num_(param.grad)
                        print(f'{name}.grad is NaN !!!')
                        param_nan += 1
                return param_nan
            def check_nan(x, name='x', title='update_traj', show=None):
                if torch.isnan(x).any():
                    print(f"{title}: {name} is NaN !!!")
                    if show is not None:
                        print(show)
                    check_param_nan()
                    return 1
                return 0
            if check_nan:
                nan_check += check_nan(policy_d, 'policy_d', show=None)
                nan_check += check_nan(expert_d, 'expert_d', show=None)
                nan_check += check_nan(log_p_policy, 'log_p_policy')
                nan_check += check_nan(log_p_expert, 'log_p_expert')
                nan_check += check_nan(log_q_policy, 'log_q_policy')
                nan_check += check_nan(log_q_expert, 'log_q_expert')
                nan_check += check_nan(gail_loss, 'gail_loss')
                nan_check += check_nan(grad_pen, 'grad_pen')
            if nan_check:
                return
            # debug end
            
            self.optimizer.zero_grad()
            (gail_loss+grad_pen).backward()
            #~ gail_loss.backward()
            #~ nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            #~ self.optimizer.step()
            # debug begin
            #~ print('check_param_nan before backward')
            if check_nan and check_param_nan():
                #~ print(torch.cat([policy_action, expert_action], dim=1))
                print(policy_action)
                return
            #~ nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            # debug end
            
            d_cosine += self.cosine_similarity(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            d_mmd += self.mmd_loss(torch.cat([policy_state, policy_action], dim=1),  torch.cat([expert_state, expert_action], dim=1))
            
        return {'loss': loss/n, 'log_p_policy': _log_p_policy/n, 'log_p_expert': _log_p_expert/n, 
                    'log_q_policy': _log_q_policy/n, 'log_q_expert': _log_q_expert/n, 
                    'd_policy': _d_policy/n, 'd_expert': _d_expert/n, 'd_cosine': d_cosine/n, 'd_mmd': d_mmd/n}
    
    def predict_energy_traj(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            energy = self.trunk(torch.cat([state, action], dim=1))
            
            reward = -energy
            
            if torch.isnan(reward).any():
                print("predict_reward_traj(): reward is NaN !!!")
                print(state)
                print(action)
                print(reward)
            
            #~ return reward
            
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_energy_traj2(self, state, action, update_rms=True):
        with torch.no_grad():
            self.eval()
            
            state = state.view(-1, self.obs_dim)
            action = action.view(-1, self.act_dim)
            
            energy = self.trunk(torch.cat([state, action], dim=1)).view(1, -1, 1)
            
            reward = torch.sum(-energy, dim=1)
            
            #~ reward = -energy
            
            if torch.isnan(reward).any():
                print("predict_reward_traj(): reward is NaN !!!")
                print(state)
                print(action)
                print(reward)
            
            return reward * 0.1
    
    def predict_reward_traj3(self, state, action):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            return reward
    
    def predict_reward_traj7(self, state, action, update_rms=True):
        with torch.no_grad():
            self.eval()
            energy = self.trunk(torch.cat([state, action], dim=1))
            
            reward = -energy
            
            if torch.isnan(reward).any():
                print("predict_reward_traj(): reward is NaN !!!")
                print(state)
                print(action)
                print(reward)
            
            return reward * 0.1
            
            if self.returns is None:
                self.returns = reward.clone()
            if update_rms:
                self.returns = reward
                self.ret_rms.update(self.returns.cpu().numpy())
            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def update_rnn_sa(self, rnn_encoder, expert_loader, rollouts, gail_batch_emb=64):
        self.train()

        loss = 0
        n = 0
        d_expert = 0
        d_policy = 0
        
        expert_batch_traj = expert_loader.dataset.sample(expert_loader.batch_size)
        
        expert_state, expert_action = expert_batch_traj
        expert_state = expert_state.to(self.device)
        expert_action = expert_action.to(self.device)
        
        expert_input = torch.cat([expert_state, expert_action], dim=2)
        expert_emb = rnn_encoder.forward_batch(expert_input, get_seq=True)
        
        expert_emb = expert_emb.reshape(-1, expert_emb.size(-1))
        expert_state = expert_state.reshape(-1, expert_state.size(-1))
        expert_action = expert_action.reshape(-1, expert_action.size(-1))
        
        expert_rollouts = feed_forward_rnn(expert_emb, expert_state, expert_action, gail_batch_emb)
        
        for expert_batch, policy_batch in zip(expert_rollouts, rollouts):
            policy_emb, policy_state, policy_action = policy_batch
            expert_emb, expert_state, expert_action = expert_batch
            
            policy_d = self.trunk(policy_emb)
            expert_d = self.trunk(expert_emb)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen_rnn(expert_emb, policy_emb)
            #~ grad_pen = self.compute_grad_pen_rnn(expert_emb.detach(), policy_emb.detach())

            loss += gail_loss.item()
            #~ loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            #~ gail_loss.backward(retain_graph=True)
            #~ gail_loss.backward()
            (gail_loss + grad_pen).backward(retain_graph=True)
            #~ (gail_loss + grad_pen).backward()
            self.optimizer.step()
            
            d_expert += self.d_score(expert_d)
            d_policy += self.d_score(policy_d)
            
        return {'loss': loss/n, 'd_expert': d_expert/n, 'd_policy': d_policy/n}
    
    def update_rnn(self, rnn_encoder, expert_loader, rollouts):
        self.train()

        loss = 0
        n = 0
        d_expert = 0
        d_policy = 0
        
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_input, policy_lengths, policy_masks = policy_batch
            
            expert_state, expert_action = expert_batch
            expert_state = expert_state.to(self.device)
            expert_action = expert_action.to(self.device)
            
            expert_input = torch.cat([expert_state, expert_action], dim=2)
            
            #~ policy_input.requires_grad = True
            #~ expert_input.requires_grad = True
            
            policy_emb = rnn_encoder.forward_batch(policy_input)
            expert_emb = rnn_encoder.forward_batch(expert_input)
            
            policy_d = self.trunk(policy_emb)
            expert_d = self.trunk(expert_emb)

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen_rnn(expert_emb, policy_emb)
            #~ grad_pen = self.compute_grad_pen_rnn(expert_emb.detach(), policy_emb.detach())

            loss += gail_loss.item()
            #~ loss += (gail_loss + grad_pen).item()
            n += 1

            rnn_encoder.prepare()
            
            self.optimizer.zero_grad()
            #~ gail_loss.backward(retain_graph=True)
            #~ gail_loss.backward()
            #~ (gail_loss + grad_pen).backward(retain_graph=True)
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
            
            rnn_encoder.update()
            
            d_expert += self.d_score(expert_d)
            d_policy += self.d_score(policy_d)
            
        return {'loss': loss/n, 'd_expert': d_expert/n, 'd_policy': d_policy/n}
    
    def predict_reward_rnn4(self, rnn_encoder, policy_input):
        with torch.no_grad():
            self.eval()
            policy_emb = rnn_encoder.forward_batch(policy_input)
            d = self.trunk(policy_emb)
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            return reward * 0.2
    
    def predict_reward_rnn(self, rnn_encoder, rnn_hxs, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            sa_emb, rnn_hxs = rnn_encoder(torch.cat([state, action], dim=1), rnn_hxs, masks)
            d = self.trunk(sa_emb)
            #~ d = self.trunk(rnn_hxs)
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8), rnn_hxs


class Discriminator0(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, lr=1e-3):
        super(Discriminator, self).__init__()

        self.device = device
        
        act_fn = nn.Tanh
        #~ act_fn = nn.ReLU
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act_fn(),
            nn.Linear(hidden_dim, hidden_dim), act_fn(),
            nn.Linear(hidden_dim, 1)).to(device)

        self.trunk.train()
        
        self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=lr)
        #~ self.optimizer = torch.optim.Adam(self.trunk.parameters(), lr=1e-4)

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())
    
    def save(self, t_start, j, reward, reward1):
        save_path = os.path.join("trained_models", time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start)))
        os.makedirs(save_path, exist_ok=True)
        model_name = f"EP_{j}_{reward:.0f}_{reward1:.0f}_d"
        torch.save({"model":self.state_dict(), "opt_state":self.optimizer.state_dict(), "ret_rms":self.ret_rms}, 
                            os.path.join(save_path, model_name + ".pt"))

    def load(self, path):
        data = torch.load(path)
        self.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt_state"])
        self.ret_rms = data["ret_rms"]
    
    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, rollouts, obsfilt=None):
        self.train()

        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_state, policy_action = policy_batch[0], policy_batch[1]
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))
            
            expert_state, expert_action = expert_batch
            if obsfilt:
                expert_state = obsfilt(expert_state.numpy()[:,:], update=False)
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(
                expert_d,
                torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(
                policy_d,
                torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            loss += (gail_loss + grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()
        return {'loss': loss / n}
    
    def update_energy(self, expert_loader, rollouts, policy=None, obsfilt=None):
        self.train()
        
        _log_p_policy = 0
        _log_p_expert = 0
        _d_policy = 0
        _d_expert = 0
        
        loss = 0
        n = 0
        for expert_batch, policy_batch in zip(expert_loader, rollouts):
            policy_state, policy_action = policy_batch[0], policy_batch[1]
            policy_lprobs = policy.eval(policy_state, policy_action)
            
            policy_d = self.trunk(
                torch.cat([policy_state, policy_action], dim=1))
            
            expert_state, expert_action = expert_batch
            expert_lprobs = policy.eval(expert_state, expert_action)
            
            if obsfilt:
                expert_state = obsfilt(expert_state.numpy()[:,:], update=False)
            
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(
                torch.cat([expert_state, expert_action], dim=1))
            
            log_p_policy = -policy_d
            log_p_expert = -expert_d
            
            log_q_policy = policy_lprobs
            log_q_expert = expert_lprobs
            #~ log_q_policy = policy_lprobs.sum(dim=1, keepdim=True)       # failed
            #~ log_q_expert = expert_lprobs.sum(dim=1, keepdim=True)       
            
            #~ log_pq_policy = torch.logaddexp(log_p_policy, log_q_policy)
            #~ log_pq_expert = torch.logaddexp(log_p_expert, log_q_expert)
            
            log_pq_policy = torch.log(log_p_policy.exp() + log_q_policy.exp())
            log_pq_expert = torch.log(log_p_expert.exp() + log_q_expert.exp())
            
            d_policy = torch.exp(log_p_policy-log_pq_policy).mean()
            d_expert = torch.exp(log_p_expert-log_pq_expert).mean()
            
            #~ log_p_tau = torch.cat([log_p_policy, log_p_expert])
            #~ log_q_tau = torch.cat([log_q_policy, log_q_expert])
            #~ log_pq = torch.cat([log_pq_policy, log_pq_expert])
            #~ d_tau = torch.exp(log_p_tau-log_pq)
            
            expert_loss = log_p_expert - log_pq_expert
            policy_loss = log_q_policy - log_pq_policy
            
            gail_loss = - expert_loss.mean() - policy_loss.mean()
            
            #~ batch_size = policy_state.size(0)
            #~ labels = torch.zeros(batch_size*2, 1)
            #~ labels[batch_size:] = 1
            
            #~ gail_loss = -torch.mean(labels*(log_p_tau-log_pq) + (1-labels)*(log_q_tau-log_pq))
            
            #~ expert_loss = F.binary_cross_entropy_with_logits(
                #~ expert_d,
                #~ torch.ones(expert_d.size()).to(self.device))
            #~ policy_loss = F.binary_cross_entropy_with_logits(
                #~ policy_d,
                #~ torch.zeros(policy_d.size()).to(self.device))

            #~ gail_loss = expert_loss + policy_loss
            
            grad_pen = self.compute_grad_pen(expert_state, expert_action,
                                             policy_state, policy_action)

            #~ loss += (gail_loss + grad_pen).item()
            loss += gail_loss.item()
            n += 1
            
            _log_p_policy += log_p_policy.mean().item()
            _log_p_expert += log_p_expert.mean().item()
            _d_policy += d_policy.item()
            _d_expert += d_expert.item()

            self.optimizer.zero_grad()
            (gail_loss+grad_pen).backward()
            #~ gail_loss.backward()
            self.optimizer.step()
        return {'loss': loss/n, 'log_p_policy': _log_p_policy, 'log_p_expert': _log_p_expert, 
                    'd_policy': _d_policy, 'd_expert': _d_expert}
    
    def predict_energy(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            energy = self.trunk(torch.cat([state, action], dim=1))
            reward = -energy
            #~ return reward
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_energy_r(self, state, action, log_p, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            energy = self.trunk(torch.cat([state, action], dim=1))
            reward = -energy
            return reward
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_reward(self, state, action, gamma, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)
    
    def predict_reward_traj(self, state, action):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)
            reward = s.log() - (1 - s).log()
            
            return reward


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(
            0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0

        self.get_idx = []
        
        for j in range(self.length):
            
            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1
            
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        return self.trajectories['states'][traj_idx][i], self.trajectories[
            'actions'][traj_idx][i], self.trajectories['obs_next'][traj_idx][i]


class ExpertDatasetTraj(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['states'].size(0)
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.trajectories['states'][i].flatten(), self.trajectories[
            'actions'][i].flatten(), self.trajectories['obs_next'][i].flatten()

class ExpertDatasetRnn(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20):
        all_trajectories = torch.load(file_name)
        
        perm = torch.randperm(all_trajectories['states'].size(0))
        idx = perm[:num_trajectories]

        self.trajectories = {}
        
        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}
        
        self.length = self.trajectories['states'].size(0)
            
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.trajectories['states'][i], self.trajectories['actions'][i]
    
    def sample(self, batch_size=64):
        indices = random.sample(range(self.length), k=batch_size)
        return self[indices]


class RolloutDatasetRnn(torch.utils.data.Dataset):
    """ pad sequence for variable-size
    Return: states, actions, lengths, masks
    """
    def __init__(self, states, actions, ep_start_idx, traj_len=5, batch_first=True):
        self.s_dim = states[0].size(1)
        self.a_dim = actions[0].size(1)
        
        self.trajectories = {'states':torch.stack(states, dim=1), 
                                    'actions':torch.stack(actions, dim=1),
                                }
        
        self.length = len(ep_start_idx)
        self.batch_first = batch_first
        self.traj_len = traj_len
        
        self.get_idx = ep_start_idx

    def __len__(self):
        return self.length

    def collate_fn_padd(self, batch):
        ## get sequence lengths
        lengths = torch.tensor([ t[2] for t in batch ])
        masks = [ torch.ones(t) for t in lengths ]
        ## padd
        states = [ torch.Tensor(t[0]) for t in batch ]
        actions = [ torch.Tensor(t[1]) for t in batch ]
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=self.batch_first)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=self.batch_first)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=self.batch_first)
        return states, actions, lengths, masks

    def __getitem__(self, i):
        j, start, steps = self.get_idx[i]
        
        states = self.trajectories['states'][j][start:start+steps]
        actions = self.trajectories['actions'][j][start:start+steps]
        
        return states, actions, steps

class RolloutDatasetRnnPack(RolloutDatasetRnn):
    """ PackedSequence
    Return: state_action, lengths, masks
    """
    
    def collate_fn_padd(self, batch):
        lengths = torch.tensor([ t[1] for t in batch ])
        masks = [ torch.ones(t) for t in lengths ]
        states = [ torch.Tensor(t[0]) for t in batch ]
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=self.batch_first)
        states = torch.nn.utils.rnn.pack_padded_sequence(states, lengths, batch_first=self.batch_first, enforce_sorted=False)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=self.batch_first)
        return states, lengths, masks

    def __getitem__(self, i):
        j, start, steps = self.get_idx[i]
        
        states = self.trajectories['states'][j][start:start+steps]
        actions = self.trajectories['actions'][j][start:start+steps]
        
        return torch.cat([states, actions], dim=1), steps

class RolloutDatasetRnnPack_test(RolloutDatasetRnn):
    """ PackedSequence
    Return: state_action, lengths, masks
    """
    
    def collate_fn_padd(self, batch):
        lengths = torch.tensor([ t[1] for t in batch ])
        masks = [ torch.ones(t) for t in lengths ]
        states = [ torch.Tensor(t[0]) for t in batch ]
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=self.batch_first)
        states = torch.nn.utils.rnn.pack_padded_sequence(states, lengths, batch_first=self.batch_first, enforce_sorted=False)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=self.batch_first)
        return states, lengths, masks

    def __getitem__(self, i):
        j, start, steps = self.get_idx[i]
        
        traj_len = 5
        steps = min(traj_len, steps)
        
        states  = torch.zeros(traj_len, self.s_dim)
        actions  = torch.zeros(traj_len, self.a_dim)
        obs_next  = torch.zeros(traj_len, self.s_dim)
        
        states[:steps] = self.trajectories['states'][j][start:start+steps]
        actions[:steps] = self.trajectories['actions'][j][start:start+steps]
        
        return torch.cat([states, actions], dim=1), steps

class RolloutDataset2(torch.utils.data.Dataset):
    """ PackedSequence
    Return: state, action, masks
    """
    def __init__(self, states, actions, obs_next, ep_start_idx, traj_len=5, batch_first=True):
        self.s_dim = states[0].size(1)
        self.a_dim = actions[0].size(1)
        
        self.trajectories = {'states':torch.stack(states, dim=1), 
                                    'actions':torch.stack(actions, dim=1),
                                }
        
        self.length = len(ep_start_idx)
        self.batch_first = batch_first
        self.traj_len = traj_len
        
        self.get_idx = ep_start_idx
    
    def __len__(self):
        return self.length
    
    def collate_fn_padd(self, batch):
        lengths = torch.tensor([ t[1] for t in batch ])
        masks = [ torch.ones(t) for t in lengths ]
        states = [ torch.Tensor(t[0]) for t in batch ]
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=self.batch_first)
        states = torch.nn.utils.rnn.pack_padded_sequence(states, lengths, batch_first=self.batch_first, enforce_sorted=False)
        masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=self.batch_first)
        return states, lengths, masks

    def __getitem__(self, i):
        j, start, steps = self.get_idx[i]
        
        steps = min(self.traj_len, steps)
        
        states  = torch.zeros(self.traj_len, self.s_dim)
        actions  = torch.zeros(self.traj_len, self.a_dim)
        
        states[:steps] = self.trajectories['states'][j][start:start+steps]
        actions[:steps] = self.trajectories['actions'][j][start:start+steps]
        
        mask = torch.ones(self.traj_len, 1)
        mask[steps:] = 0
        
        return states.flatten(), actions.flatten(), mask.bool()

class RolloutDataset(torch.utils.data.Dataset):
    """ Simple trajectory
    """
    def __init__(self, states, actions, obs_next, ep_start_idx, ep_in_bounds={}, ep_theta={}, ep_mobj={}, traj_len=5):
        self.s_dim = states[0].size(1)
        self.a_dim = actions[0].size(1)
        
        self.trajectories = {'states':torch.cat(states, dim=1), 
                                    'actions':torch.cat(actions, dim=1),
                                    'obs_next':torch.cat(obs_next, dim=1),
                                }
        
        self.traj_len = traj_len
        self.length = len(ep_start_idx)
        
        self.get_idx = ep_start_idx
        self.ep_in_bounds = ep_in_bounds
        self.ep_theta = ep_theta
        self.ep_mobj = ep_mobj
        
        #~ random.shuffle(self.get_idx)
    
    def save(self, t_start, j, reward, reward1):
        save_path = os.path.join("trained_models", time.strftime('%Y%m%d_%H%M%S', time.localtime(t_start)))
        os.makedirs(save_path, exist_ok=True)
        model_name = f"EP_{j}_traj_{reward:.0f}_{reward1:.0f}"
        self.trajectories['ep_start_idx'] = self.get_idx
        self.trajectories['ep_in_bounds'] = self.ep_in_bounds
        self.trajectories['ep_theta'] = self.ep_theta
        self.trajectories['ep_mobj'] = self.ep_mobj
        self.trajectories['traj_len'] = self.traj_len
        self.trajectories['length'] = self.length
        self.trajectories['s_dim'] = self.s_dim
        self.trajectories['a_dim'] = self.a_dim
        torch.save(self.trajectories, os.path.join(save_path, model_name + ".pt"))
    
    def load(self, save_path):
        self.trajectories = torch.load(save_path)
        self.get_idx = self.trajectories['ep_start_idx']
        self.ep_in_bounds = self.trajectories['ep_in_bounds']
        self.ep_theta = self.trajectories['ep_theta']
        self.ep_mobj = self.trajectories['ep_mobj']
        self.traj_len = self.trajectories['traj_len']
        self.length = self.trajectories['length']
        self.s_dim = self.trajectories['s_dim']
        self.a_dim = self.trajectories['a_dim']

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        j, start, steps = self.get_idx[i]
        steps = min(self.traj_len, steps)
        
        states  = torch.zeros(self.traj_len*self.s_dim)
        actions  = torch.zeros(self.traj_len*self.a_dim)
        obs_next  = torch.zeros(self.traj_len*self.s_dim)
        
        #~ states  = torch.randn(self.traj_len*self.s_dim)
        #~ actions  = torch.randn(self.traj_len*self.a_dim)
        #~ obs_next  = torch.randn(self.traj_len*self.s_dim)
        
        #~ states  = torch.ones(self.traj_len*self.s_dim)
        #~ actions  = torch.ones(self.traj_len*self.a_dim)
        #~ obs_next  = torch.ones(self.traj_len*self.s_dim)
        
        # todo: error, start = start*self.s_dim !!!
        start_s = start*self.s_dim
        start_a = start*self.a_dim
        states[:steps*self.s_dim] = self.trajectories['states'][j][start_s:start_s+steps*self.s_dim]
        actions[:steps*self.a_dim] = self.trajectories['actions'][j][start_a:start_a+steps*self.a_dim]
        #~ obs_next[:steps*self.s_dim] = self.trajectories['obs_next'][j][start:start+steps*self.s_dim]
        
        #~ return states, actions, obs_next
        
        mask = torch.ones(self.traj_len, 1)
        mask[steps:] = 0
        
        return states, actions, mask.bool()
        
        #~ return states, actions


class RnnEncoder(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lr=1e-3):
        super().__init__()
        
        input_dim = obs_dim + act_dim
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    
    def prepare(self):
        self.optimizer.zero_grad()
    
    def update(self):
        self.optimizer.step()
    
    def forward(self, x, hxs, masks):
        """ batch, len, size """
        x, hxs = self.gru(x.unsqueeze(1), (hxs * masks).unsqueeze(0))
        return x.squeeze(1), hxs.squeeze(0)
    
    def forward_batch(self, x, get_seq=False):
        """ (B,T,N) or (B,N) """
        x, hxs = self.gru(x)
        return x if get_seq else hxs.squeeze(0)

