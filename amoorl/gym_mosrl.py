import gym
from gym import spaces
from gym.utils import seeding

#~ import torch
import math
import numpy as np
from numpy import linalg as LA
import copy
import collections as col
import os
import time
import random

from sklearn import preprocessing
from sklearn import svm
#~ from sklearn.externals import joblib
import joblib

import pygmo as pg

from math import pi as PI

from .zfilter import ZFilter
from .xfilter import XFilter


def make_env_mosrl(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        debug = env_args.get('debug', 0)
        #
        env = MosrlEnv(id=rank, debug=debug, )
#         env = MooEnv2(id=rank, debug=debug, )
        # Benchmark Begin
        #~ env = MooEnvX(id=rank, debug=debug, )
        # Benchmark End
        env.seed(seed + rank)
        return env
    return _thunk

def make_env_mosrl2(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        debug = env_args.get('debug', 0)
        #
        env = MosrlEnv2(id=rank, debug=debug, )
        env.seed(seed + rank)
        return env
    return _thunk


def MinMaxScale(x, x_min, x_max, clip=1):
    x = np.array(x)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    xn = (x - x_min)/(x_max - x_min)
    if not clip:
        return xn
    return np.clip(xn, 0, 1)


class MosrlEnv(object):
    """ Multi-objective optimization
    """
    def __init__(self, id=0, debug=1, obs_type=0, **kwds):
        self.id = id
        self.debug = self.id==0 and debug
        self.obs_type = obs_type
        self.moo = 0
#         self.moo = 1
#         self.moo = 2
#         self.moo = 3
        
        self.shared_pop_x = None
        self.moo_flag = [0]
        
#         self.pretrain = 1
        self.pretrain = 0
        
        #~ self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32 )
        
        if self.moo==1:
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(200+6,), dtype=np.float32 )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2+6+2+6,), dtype=np.float32 )
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6+2+6,), dtype=np.float32 )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6+2,), dtype=np.float32 )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32 )
        elif self.moo==2:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32 )
        elif self.moo==3:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6+1,), dtype=np.float32 )
        elif 1:
#         if self.obs_type==0:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32 )
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32 )
        
        self.init_mop()
        
        # debug begin
        self.frame_count = 0
        self.total_steps = 0
        # debug end
        self._gym_vars()
    
    def set_DK(self, D, K):
        self.D = D
        self.K = K
        self.Dfinal = self.D - self.K*2
    
    def init_mop(self):
        _path = os.path.dirname(__file__)
#         self.estimator = joblib.load(os.path.join(_path, './model_svr.pkl'))
        self.estimator = joblib.load(os.path.join(_path, './model_gp.pkl'))
        
#         self.D = 60
#         self.Dfinal = 30
        
        self.D = 60
        self.Dfinal = self.D - 10*2
        
        self.K = (self.D - self.Dfinal)/2
        
        self.fVc = None
        self.fF = None
        self.fAp = None
        
        self.rVc = None
        self.rF = None
        self.rAp = None
        
        self.LimitVc = 14, 240
        self.LimitF = 0.06, 0.3246
        self.LimitApr = 1.0, 2.75
        self.LimitApf = 0.5, 1.0
        
        rngVc = self.LimitVc[1] - self.LimitVc[0]
        rngF = self.LimitF[1] - self.LimitF[0]
        rngApf = self.LimitApf[1] - self.LimitApf[0]
        rngApr = self.LimitApr[1] - self.LimitApr[0]
        rngMax = max(rngVc, rngF, rngApf, rngApr)
        self.ActFactor = np.array([rngVc/rngMax, rngF/rngMax, rngApf/rngMax,
                                              rngVc/rngMax, rngF/rngMax, rngApr/rngMax,])
        
        self.Limit3f = zip(self.LimitVc, self.LimitF, self.LimitApf)
        self.Limit3f = list(map(np.array, self.Limit3f))
        self.Limit3r = zip(self.LimitVc, self.LimitF, self.LimitApr)
        self.Limit3r = list(map(np.array, self.Limit3r))
        self.Limit3 = zip(self.LimitVc, self.LimitF, [self.LimitApf[0], self.LimitApr[1]])
        self.Limit3 = list(map(np.array, self.Limit3))
        
        self.Limit6 = zip(self.LimitVc, self.LimitF, self.LimitApf, self.LimitVc, self.LimitF, self.LimitApr)
        self.Limit6 = list(map(np.array, self.Limit6))
        self.LimitPcut = 1e-8, 4000.
        
        self.Vminmax = 14, 247
        self.Fminmax = 0.06, 0.3246
        self.APminmax = 0.5, 2.75
#         self.Vminmax = 100, 247.1651
#         self.Fminmax = 0.1, 0.25
#         self.APminmax = 0.7, 2
        
        self.VFAPminmax = list(zip(self.Vminmax, self.Fminmax, self.APminmax))
        
        self.obj_prev = None
        self.solution_info = None
        self.best_reward = None
        
        self.best_sol = []
        self.best_sol_start = 0
        
        self.pop = None
        self.pop_x = None
        self.N = 100
        
        self.step_objs = np.zeros(2)
        self.step_objs_dt = np.zeros(2)
        
        self.sol = np.zeros(self.action_space.shape[0])
        self.sol_dt = np.zeros(self.action_space.shape[0])
        
        self.min_max_scaler = preprocessing.MinMaxScaler()
    
    def update_pop(self, x, y, select=0):
        self.pop_x = np.vstack((self.pop_x, x))
        self.pop = np.vstack((self.pop, y))
        if select:
            self.selectN(self.N)
    
    def selectN(self, N):
        if self.moo:
            indices = pg.select_best_N_mo(self.pop, N)
        else:
            indices = np.argsort(self.pop.ravel()) [:N]
        self.pop_x = self.pop_x[indices]
        self.pop = self.pop[indices]
    
    def cross_over(self):
        self.selectN(self.N)
        X = np.random.permutation(self.pop_x.ravel()).reshape(self.pop_x.shape)
        noise = np.random.randn(*self.pop_x.shape)*1e-1
        X1 = (self.pop_x + noise).clip(self.action_space.low, self.action_space.high)
        X = np.vstack((X, X1))
        y = self.eval_f(X)
        self.update_pop(X, y)
        self.selectN(self.N)
    
    def eval_f(self, X, act_type=1):
        y = []
        for action in X:
            self.on_action(action, type=act_type)
            cuts = self.planning_cuts()
            f0, f1, f2, p0, info = self.f_objectives(cuts)
            objs = np.array([f0, f1, f2])
            # penalty begin
            p_ap = self.penaltyApr(cuts[-2])
            p_3 = self.penalty_3()
            punish = p_ap + p_3
            objs += punish
            # penalty end
            y.append(objs[:2])
        return y
    
    def get_states(self):
        return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
    
    def make_obs(self, obj=0, randn=0):
        if randn:
           return np.random.randn(self.observation_space.shape[0])
#         elif obj:
#             return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp, self.obj_prev[obj]])
        if self.moo==1:
            obs_soo = self.get_states()
#             obs_moo = self.make_obs_moo()
#             obs = np.concatenate((obs_moo, obs_soo))
#             obs = np.concatenate((np.log(self.step_objs), obs_soo))
#             obs = np.concatenate((np.log(self.step_objs), np.tanh(self.step_objs_dt), self.sol_dt, obs_soo))
            obs = np.concatenate((np.tanh(self.step_objs_dt), self.sol_dt, obs_soo))
#             obs = np.concatenate((np.tanh(self.step_objs_dt), obs_soo, ))
#             obs = obs_soo
            return obs
        elif self.moo==2:
            return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
        elif self.moo==3:
#             self.alpha = np.random.uniform(0, 1)
            return np.array([self.alpha, self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
        return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
    
    def make_obs_moo(self, ):
        if self.pop is None or self.pop.size<200:
            return np.random.uniform(0, 1, size=200)
        obs = self.min_max_scaler.fit_transform(self.pop)
        return obs.ravel()
    
    def set_states(self, fVc, fF, fAp, rVc, rF, rAp):
        """ debug only """
        self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp = fVc, fF, fAp, rVc, rF, rAp
    
    def init_states(self, ):
        # pre-train begin
        if self.pretrain:
            d = np.random.uniform(55, 65)
            k = np.random.uniform(d/4, 0.9*d/2)
            self.D = d
            self.Dfinal = k
        # pre-train end
        
        # moo3
        self.alpha = np.random.uniform(0, 1)
        
        if not self.moo and self.total_steps>1e4 and random.random()<0.7:
            self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp = self.best_sol
            self.best_sol_start = 1
        elif self.moo==2 and self.moo_flag[0] and random.random()<0.5 and 1:
            action = random.choice(self.shared_pop_x.numpy())
            self.on_action(action, type=1)
            self.best_sol_start = 2
        elif self.moo==1 and self.pop_x is not None and random.random()<0.5 and 1:
            action = random.choice(self.pop_x)
            self.on_action(action, type=1)
            self.best_sol_start = 2
        else:
            self.best_sol_start = 0
            
            self.fVc = np.random.uniform(*self.LimitVc)
            self.fF = np.random.uniform(*self.LimitF)
            self.fAp = np.random.uniform(*self.LimitApf)
            
            self.rVc = np.random.uniform(*self.LimitVc)
            self.rF = np.random.uniform(*self.LimitF)
            LimitApr = self.LimitApr[1] - self.fAp - np.finfo(np.float32).eps
            assert self.LimitApr[0] <= LimitApr
            self.rAp = np.random.uniform(self.LimitApr[0], LimitApr)
        # initial eval begin
        action = np.zeros(self.action_space.shape[0])
        fitness = self.eval_f([action], act_type=0)
        self.step_objs = fitness[0]
        # initial eval end
    
    def check_states(self, obs):
        if (self.Limit6[0]>obs).any() or (self.Limit6[1]<obs).any():
            print('!check_states failed!')
            return False
        return True
    
    def planning_cuts(self, eps=.05):
        """ k: 初始余量 (mm)
        """
        k = (self.D - self.Dfinal)/2
        #~ assert k>0
        rAp0, fAp = self.rAp, self.fAp
        rApL = (k - fAp)%rAp0
        #~ assert 0<=rApL and rApL<=rAp0
        d = rApL / rAp0
        if d<=eps:
            rApL += rAp0
            Nr = (k - fAp) // rAp0
        else:
            rApL = (k - fAp) % rAp0
            Nr = (k - fAp) // rAp0 + 1
        cuts = [rAp0,]*int(Nr-1) + [rApL, fAp]
        #~ assert Nr>1 and min(cuts)>0
        # debug begin
        #~ if not (np.array(cuts)>0).all():
            #~ print(len(cuts), min(cuts))
            #~ print(cuts)
        # debug end
        return cuts
    
    def Pcut_old(self, cuts):
        """ x_fix: al,40cr,45,80,55,steel
        """
#         Xs, x_fix = [], [0,1,0,0,1,0]
        Xs, x_fix = [], [0,1,0,1,0,0]
#         Xs, x_fix = [], [1,0,0,1,0,0]
        for rAp in cuts[:-1]:
            rX = MinMaxScale([self.rVc, self.rF, rAp], *self.VFAPminmax)
            Xs.append( np.append(rX, x_fix) )
        fX = MinMaxScale([self.fVc, self.fF, cuts[-1]], *self.VFAPminmax)
        Xs.append( np.append(fX, x_fix) )
        Xs = np.array(Xs)
#         Xs = np.array(Xs) [:,:3]
        Pc = self.estimator.predict(Xs).clip(*self.LimitPcut)
        # debug begin
        #~ print(Xs.shape, Xs)
        #~ print('Pc', Pc.shape)
        #~ for p in Pc:
            #~ print('\t%.3f'%p)
        # debug end
        return Pc
    
    def Pcut(self, cuts):
        """ x_fix: al,40cr,45,80,55,steel
        """
#         x_fix = [0,1,0,0,1,0]
        x_fix = [0,1,0,1,0,0]
#         x_fix = [1,0,0,1,0,0]
        
        rAp = np.asarray(cuts[:-1]).reshape(-1,1)
        rX = [self.rVc, self.rF]
        if rAp.shape[0]>1:
            rX = np.resize(rX, (rAp.shape[0], 2))
        rX = np.hstack((rX, rAp))
        rX = MinMaxScale(rX, *self.VFAPminmax)
        if rX.ndim>1:
            rx_fix = np.resize(x_fix, (rX.shape[0], 6))
        else:
            rx_fix = x_fix
        Xr = np.hstack((rX, rx_fix))
        
        fX = MinMaxScale([self.fVc, self.fF, cuts[-1]], *self.VFAPminmax)
        if fX.ndim>1:
            fx_fix = np.resize(x_fix, (fX.shape[0], 6))
        else:
            fx_fix = x_fix
        Xf = np.hstack((fX, fx_fix))
        
        Xs = np.vstack((Xr, Xf))
#         Xs = Xs[:,:3]
        Pc = self.estimator.predict(Xs).clip(*self.LimitPcut)
        # debug begin
        #~ print(Xs.shape, Xs)
        #~ print('Pc', Pc.shape)
        #~ for p in Pc:
            #~ print('\t%.3f'%p)
        # debug end
        return Pc
    
    def f_objectives(self, cuts, Lair=15, Lcut=100, Pst=945, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径,所以d_f是self.D减去粗加工的剩余
        """
        Ds, apx2 = [self.D], 0
        for ap in cuts[:-1]:
            apx2 += ap*2
            Ds.append(self.D - apx2)
        Dr, d_f = Ds[:-1], Ds[-1]
        # debug begin
        #~ print('cuts', cuts)
        #~ print('Ds', Ds)
        # debug end
        # debug info
        info = {}
        # debug end
        
        def t(D, L, Vc, f):return 60* 3.14*D*L/(1000.*Vc*f)
        
        t_air_r = [t(d_r, Lair, self.rVc, self.rF) for d_r in Dr]
        t_air_f = t(d_f, Lair, self.fVc, self.fF)
        t_cut_r = [t(d_r, Lcut, self.rVc, self.rF) for d_r in Dr]
        t_cut_f = t(d_f, Lcut, self.fVc, self.fF)
        # debug begin
        info.update(t_cut_r=t_cut_r, t_cut_f=t_cut_f)
        # debug end
        # debug begin
        #~ print('t_air_r')
        #~ print_list(t_air_r)
        #~ print('t_air_f %.3f'%t_air_f)
        #~ print('t_cut_r')
        #~ print_list(t_cut_r)
        #~ print('t_cut_f %.3f'%t_cut_f)
        # debug end
        
        def Est():
            return Pst*(t_st + sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        def Eu():
#             def pu(Vc,D):return -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
#             def pu(Vc,D):return 1.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
#             def pu(Vc,D):return 0.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
            def pu(Vc,D):return 3.079*(1e3*Vc/(PI*D)) + 0.013*(1e3*Vc/(PI*D))**2 + 92.9
            Pu_r = np.array( [pu(self.rVc, d_r) for d_r in Dr] )
            Pu_f = pu(self.fVc, d_f)
            return Pu_r.dot(t_air_r) + Pu_f*t_air_f + Pu_r.dot(t_cut_r) + Pu_f*t_cut_f
        
        def Emr():
            Pmr = self.Pcut(cuts)
            # debug begin
            info.update(Pmr=Pmr)
            # debug end
            return Pmr[:-1].dot(t_cut_r) + Pmr[-1]*t_cut_f
        
        def Eauc(Pcf=80, Phe=1000):
            return (Pcf+Phe)*(sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        #~ def T(Vc, f, Ap):return 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.33*Ap**0.28)
#         def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**0.95*Ap**0.33)
        def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.4*f**0.95*Ap**0.33)
        
        Tr = [T(self.rVc, self.rF, ap_r) for ap_r in cuts[:-1]]
        Tf = T(self.fVc, self.fF, cuts[-1])
        t_ect = np.array(t_cut_r).dot(1/np.array(Tr)) + t_cut_f/Tf
        # debug begin
        #~ print('\nTr', len(Tr))
        #~ for tr_ in Tr:
            #~ print('\t%.3f'%tr_)
        #~ print('Tf %.3f\n'%Tf)
        # debug end
        
        def Ect():
            #~ return Pst*t_pct*t_ect
            return (Pst*t_pct+5340185.)*t_ect
#             return (Pst*t_pct+0.)*t_ect
        
        # debug begin
        e_st = Est()
        e_u = Eu()
        e_mr = Emr()
        e_auc = Eauc()
        e_ct = Ect()
        #~ print('t_cut_r vs Dr')
        #~ for t_r, d_r in zip(t_cut_r, Dr):
            #~ print('\t%.3f: %.3f'%(t_r, d_r))
        #~ print('f\t%.3f: %.3f\n'%(t_cut_f, d_f))
        #~ print('e_st: %.3f'% e_st)
        #~ print('e_u: %.3f'% e_u)
        #~ print('e_mr: %.3f'% e_mr)
        #~ print('e_auc: %.3f'% e_auc)
        #~ print('e_ct: %.3f'% e_ct)
        #~ print('Esum: %.3f'% (e_st+e_u+e_mr+e_auc+e_ct) )
        SEC = (e_st+e_u+e_mr+e_auc+e_ct)/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        # debug end
        #~ SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + sum(t_air_r) + sum(t_cut_r) + t_air_f + t_cut_f + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5, N=4):
            kt = Ch/400 + Ci/(0.75*ne)
            #~ return k0*Tp + ke*Tp + kt*t_ect
            return k0*Tp/60 + ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6 + kt*t_ect/N
        
        # debug begin
        info.update(e_st=e_st, e_u=e_u, e_mr=e_mr, e_auc=e_auc, e_ct=e_ct)
        # debug end
        
        return SEC, Tp, Cp(N=3), self.penaltyRa(), info
    
    def penaltyRa(self, k=0, u=1, a=2, Ramax=1.6, Kmax=10):
        g = self.rF**2/(8*0.8) - Ramax
        p = a**min(k, Kmax) *u* max(0, g)**2
        return p
    
    def penaltyApr(self, ap_r, a=1e6, k=2):
        """ default a=10 """
        gmin = self.LimitApr[0] - ap_r    # min < ap_r
        gmax = ap_r - self.LimitApr[1]    # ap_r < max
        pmin = max(0, gmin)**k
        pmax = max(0, gmax)**k
        return a*(pmin + pmax)
    
#     def penalty_3(self, a=1e6, t=2, k1=1, k2=2.5, k3=1):
    def penalty_3(self, a=1e6, t=2, k1=2.5, k2=1.5, k3=1):
        """ fVc > k1 * rVc   k1 = 1.
        rF  > k2 * fF    k2 = 2.5
        rAp > k3 * fAp   k3 = 1.
        """
        g_fVc = k1 * self.rVc - self.fVc
        g_rF = k2 * self.fF - self.rF
        g_rAp = k3 * self.fAp - self.rAp
        ci1 = max(0, g_fVc)**t
        ci2 = max(0, g_rF)**t
        ci3 = max(0, g_rAp)**t
#         return a*(ci1+ci2+ci3)
        return a*(ci2+ci3)
    
    def function(self, action, mo=0, eval=0):
        """ For SwarmPackagePy """
        #~ print(type(action), action)
        self.on_action(action, type=1)
        cuts = self.planning_cuts()
        f0, f1, f2, p0, info = self.f_objectives(cuts)
        mobj = np.array([f0, f1, f2])
        # constraints begin
        punish = self.penaltyApr(cuts[-2])
        punish += self.penalty_3()
        mobj += punish
        # constraints end
        obj = f0
        if eval:
            info.update(f0=f0, f1=f1, f2=f2, punish=punish)
            return self.make_obs(), np.array(cuts), info
        if mo:
            return mobj
        return obj
    
    def reward_hv(self, f_objs, reward=0):
        if self.pop is None:
            self.pop = np.array( [f_objs[:2]] )
            self.pop_x = np.array( [self.get_states()] )
        else:
            test_p = np.vstack( (self.pop, f_objs[:2]) )
            hv = pg.hypervolume(test_p)
            reward = hv.exclusive(len(self.pop), hv.refpoint(offset=0.1))
            if reward>0:
#                 indices = pg.select_best_N_mo(test_p, 100)
#                 self.pop = test_p[indices]
#                 self.update_pop(self.sol, f_objs[:2], select=1)
                self.update_pop(self.sol, f_objs[:2])
                reward = math.tanh(reward)
        return reward
    
    def reward_hv1(self, f_objs, reward=0):
        if self.pop is None:
            self.pop = np.array( [f_objs[:2]] )
        else:
            test_p = np.vstack( (self.pop, f_objs[:2]) )
            indices = pg.select_best_N_mo(test_p, 100)
            self.pop = test_p[indices]
            # minimax begin
#             min_max_scaler = preprocessing.MinMaxScaler()
            f = self.min_max_scaler.fit_transform(self.pop)
            # minimax end
            hv = pg.hypervolume(f)
            reward = hv.compute(hv.refpoint(offset=0.1))
#             reward = math.log(reward)
#             reward = math.tanh(reward)
        return reward
    
    def reward(self, alpha=0):
        """ Normalized reward """
        info = {}
        cuts = self.planning_cuts()
        f0, f1, f2, p0, info = self.f_objectives(cuts)
        objs = np.array([f0, f1, f2])
        #~ objs += p0
        # constraints begin
#         punish = self.penaltyApr(cuts[-2])
#         punish += self.penalty_3()
#         objs += punish
        # constraints end
        if self.obj_prev is None:
            self.obj_prev = objs.copy()
        Rs = self.obj_prev - objs
#         Rs = np.sign(Rs)
        Rs = np.tanh(Rs)
        self.obj_prev = objs.copy()
        # constraints begin
        p_ap = self.penaltyApr(cuts[-2])
        p_3 = self.penalty_3()
        punish = p_ap + p_3
        objs += punish
#         Rs += -punish
#         Rs += -np.tanh(punish)
        Rs += -np.tanh(p_ap) -np.tanh(p_3)
        # constraints end
        step_objs = objs.copy() [:2]
        self.step_objs_dt = step_objs - self.step_objs
        self.step_objs = step_objs
        if self.moo==2 or self.moo==3:
            P = np.tanh(p_ap) + np.tanh(p_3)
            if P>0:
                R = -P
            else:
                R = alpha* 10/f0 + (1-alpha)* 500/f1
            info.update(punish=punish, mobj=objs)
        elif self.moo==1:
            R = self.reward_hv(objs.copy())
            R += -np.tanh(p_ap) -np.tanh(p_3)
            info.update(punish=punish, mobj=objs)
        else:
            R = Rs[self.obs_type]
#         R = np.tanh(Rs[0])
        # debug begin
#         R = np.tanh(-objs[0])
#         R = -f0
#         R = -(f0 + punish)
#         R = -np.tanh(f0 + punish)
        # debug end
        return R, objs[self.obs_type], cuts, info
#         return R, f0, cuts
    
    def _action_scale(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        act_k = (self.Limit6[1] - self.Limit6[0])/ 2.
        act_b = (self.Limit6[1] + self.Limit6[0])/ 2.
        return act_k * action + act_b
    
    def on_action(self, action, type=0):
        """ States 
        """
        #~ d_fVc, d_fF, d_fAp, d_rVc, d_rF, d_rAp = np.clip(action, *self.Limit6)
        
        #~ self.fVc += d_fVc
        #~ self.fF += d_fF
        #~ self.fAp += d_fAp
        
        #~ self.rVc += d_rVc
        #~ self.rF += d_rF
        #~ self.rAp += d_rAp
        
        #~ self.fVc, self.fF, self.fAp, \
        #~ self.rVc, self.rF, self.rAp = self.make_obs().clip(*self.Limit6)
        
        #~ action = self._action_scale(action)
        
        if type==0:
#             action = action * self.ActFactor * 10
#             action = action * self.ActFactor * 3
            action = action * self.ActFactor * 3.5
            self.fVc, self.fF, self.fAp, \
            self.rVc, self.rF, self.rAp = (self.get_states()+action).clip(*self.Limit6)
        else:
            self.fVc, self.fF, self.fAp, \
            self.rVc, self.rF, self.rAp = self._action_scale(action).clip(*self.Limit6)
        
        sol = self.get_states() / (self.Limit6[1]-self.Limit6[0])
        self.sol_dt = sol - self.sol
        self.sol = sol
        # debug begin
        #~ if self.debug:
            #~ print(self.make_obs())
        # debug end
    
    def reset(self, ):
        self.init_states()
        self.reward()
        self.frame_count = 0
        # Generator Begin
#         return self.make_obs(randn=1)
        # Generator End
        return self.make_obs(obj=self.obs_type)
    
    def step(self, action):
        
        info, done = {}, 0
        
        alpha = 0
        if self.moo==2:
            alpha = action[0].clip(0, 1)
            action = action[1:]
        elif self.moo==3:
            alpha = self.alpha
        
        self.on_action(action, type=0)    # current
#         self.on_action(action, type=1)
#         self.on_action(np.random.uniform(-1, 1, action.shape), type=1)
        
        reward, f0, cuts, step_info = self.reward(alpha)
        
        #~ if self.check_states(obs) is False:
            #~ done = 1
            #~ reward = -1
            #~ print("Fatal Error: should not happen")
        
        obs = self.make_obs(obj=self.obs_type)
        # Generator Begin
        #obs = self.make_obs(randn=1)
        # Generator End
        
        # debug begin
        def get_solution():
            return 'Nc:%d f(Vc%.3f F%.3f Ap%.3f) r(Vc%.3f F%.3f Ap%.3f) obj:%.3f'%\
                    (len(cuts), self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp, f0)
        if self.solution_info is None or f0<self.solution_info[1]:
            solution = get_solution()
            self.solution_info = [solution, f0, 1]
            if self.best_reward is None:
                self.best_reward = f0+1
        #~ if self.debug:
            #~ print('\ncuts %rmm %r reward %.3f'%(sum(cuts), len(cuts), reward))
            #~ print(self.solution_info)
        info.update(f0=f0)
        
        self.total_steps += 1
        # limit steps begin
        self.frame_count += 1
        if self.frame_count>=1000:
#         if self.frame_count>=500:
            self.frame_count = 0
            done = 1
        # limit steps end
        
        if done:
            if self.debug:
                print('best_sol_start %d, total_steps %d'%(self.best_sol_start, self.total_steps) )
                print('done', get_solution())
                print('best', self.solution_info)
            if self.best_reward>self.solution_info[1]:
                info.update(solution=self.solution_info)
                self.best_reward = self.solution_info[1]
                self.best_sol = [self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp]
            if self.moo==1 and 1:
                self.cross_over()
        
        if self.moo:
            info.update(pref=[50,1000])
            info.update(penalty=step_info['punish'])
            if step_info['punish']<1e-6:
                info.update(mobj=step_info['mobj'].copy()[:2], pop_x=self.get_states())
        
        return obs, float(reward), done, info
    
    
    # ----- Rainbow Dqn Special -----
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def action_space_n(self):
        return self.action_space.n
    
    def close(self):
        pass
    
    # ----- Gym Special -----
    def _gym_vars(self):
        self.seed()
        self._spec = None
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.reward_range = (-100.0, 100.0)
        self.repeat_action = 0
    
    @property
    def spec(self):
        return self._spec

    @property
    def unwrapped(self):
        return self
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        """ show something
        """
        return None


class MosrlEnv2(MosrlEnv):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        
#         if 0:
        if 1:
            self.single_obj_type = 0
        # multi task begin
        elif self.id%2==0:
            self.single_obj_type = 0
        else:
            self.single_obj_type = 1
        # multi task end
        self.moo = 0
#         self.moo = 1
#         self.pretrain = 1    # random task
        self.pretrain = 0
#         self.pretrain = 2   # mtl mode
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32 )
        
        if self.moo:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5+2,), dtype=np.float32)
        else:
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,) )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,) )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32 )
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,) )
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,) , dtype=np.float32)
#             self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,) )
        
        # best of lifetime
        self.best_solution = dict(obj=None, sol=[])
#         self.ref_point = [25, 25000]

        self.ref_point = [101, 2001] if self.pretrain else [41, 1001]
        self.ref_clip = [100, 2000] if self.pretrain else [40, 1000]
        
#         self.ref_point = [101, 2001]
#         self.ref_clip = [100, 2000]
        
        self.ep_objs = np.array([0.,0.,0.])
        self.ep_objs_prev = self.ep_objs.copy()
        self.ep_count = 0
        
        self.pop = None
        self.moo_pop = None
        self.moo_flag = [0]
        
#         self.zf_objs = ZFilter(3, demean=False)
#         self.zf_penalty = ZFilter(3, demean=False)
        
#         self.zf_objs = ZFilter(1)
        self.zf_objs = XFilter(1)
        self.zf_penalty = XFilter(1)
        self.zf_penalty1 = XFilter(1)
        
        self.init_phr()
    
    def reset_task(self, task={}):
        D = task.get('D', 60)
        K = task.get('K', 30)
        self.set_DK(D, K)
        return self.reset()
    
    def init_states(self, ):
        def rand_DK():
            d = np.random.uniform(55, 60)   # Flexible 1
#             d = np.random.uniform(50, 65)   # Flexible 2
            k = np.random.uniform(d/4, 0.9*d/2)
#             k = np.random.uniform(12, 17)
            return d, k
        # pretrain begin
        
        if self.pretrain==1:
            self.set_DK(*rand_DK())
        # controlled by reset_task()
        elif self.pretrain==2:
            pass
        # pretrain end
        else:
#             self.set_DK(self.D, self.K)
            #~ self.set_DK(66, 20)
#             self.set_DK(65, 15)
            self.set_DK(45, 10)     # case1 ea
            #~ self.set_DK(45, 6)     # case2 ea
        
        self.Dt = self.D
        self.Kt = self.K
        
        self.D0 = self.Dt
        self.K0 = self.Kt
        self.Stage = 0
        self.StageFin = 0
        self.StageId = 0
#         self.tVc = None
#         self.tF = None
#         self.tAp = None
        self.tVc = -1
        self.tF = -1
        self.tAp = -1
        
        self.rVc_max = self.LimitVc[0]
        self.rF_min = self.LimitF[1]
        self.rAp_min = self.LimitApr[1]
        
        self.ep_solution = dict(obj=None, sol=[])
        
        self.ep_objs_prev = self.ep_objs.copy()
        self.ep_objs = np.array([0.,0.,0.])
        self.ep_punish = 0
        self.ep_R = 0
        
        self.step_punish = 0
#         self.step_penalty = np.zeros(4)
        self.step_penalty = np.zeros(3)
        
        self.step_objs = np.array([0.,0.,0.])
    
    def update_moo_pop(self):
#         if self.moo_pop is not None and self.moo_flag[0]:
        if self.moo_flag[0]:
            self.pop = self.moo_pop.numpy().copy()
    
    # TODO: nothing
    def make_obs(self):
#         return np.array( [self.Stage, self.Dt, self.Kt] )
        if self.moo:
#             return np.array( [*self.ep_objs_prev[:2], self.Stage, self.Dt, self.Kt] )
#             return np.array( [*self.step_objs[:2], self.Stage, self.Dt, self.Kt] )
#             return np.array( [*np.log(self.step_objs[:2]+1), self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )  # pretrain test
            return np.array( [self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )  # pretrain test
#             return np.array( [self.Dt, self.Kt, self.tVc,self.tF,self.tAp, *np.log(1+self.step_objs[:2]+self.step_punish)] )  # pretrain test
        else:
#             return np.array( [self.step_objs[self.single_obj_type], self.Stage, self.Dt, self.Kt] )
#             return np.array( [self.Stage, self.Dt, self.Kt] )
#             return np.array( [self.StageId, self.Dt, self.Kt] )
#             return np.array( [self.Kt] )
#             return np.array( [self.Dt, self.Kt] )
#             return np.array( [self.Dt/self.D, self.Kt/self.K0] )
#             return np.array( [self.step_objs[self.single_obj_type], self.Dt, self.Kt] )
#             return np.array( [self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )  # pretrain test
#             return np.array( [self.StageId, self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )  # pretrain test
            return np.array( [self.StageId, self.Dt, self.Kt] )  # pretrain test
#             return np.array( [self.StageId] )  # pretrain test
#             return np.array( [self.step_objs[self.single_obj_type], self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )  # pretrain
#             return np.array( [self.step_objs[self.single_obj_type], *self.step_penalty, self.Dt, self.Kt, self.tVc,self.tF,self.tAp] )
#             return np.array( [self.ep_punish, self.Dt, self.Kt] )
#             return np.array( [self.step_objs[self.single_obj_type], self.ep_punish, self.Dt, self.Kt] )
#             return np.array( [self.Stage, self.step_objs[self.single_obj_type], *self.step_penalty, self.Dt, self.Kt] )
#             return np.array( [self.Stage, self.step_objs[self.single_obj_type], self.Dt, self.Kt] )
#             return np.array( [self.ep_objs[self.single_obj_type], self.Dt, self.Kt] )
    
    def Pcut(self):
        """ x_fix: al,40cr,45,80,55,steel
        """
#         x_fix = [0,1,0,0,1,0]
        x_fix = [0,1,0,1,0,0]   # 2020 used
#         x_fix = [1,0,0,1,0,0]
        X = MinMaxScale([[self.tVc, self.tF, self.tAp]], *self.VFAPminmax)
        if X.ndim>1:
            x_fix = np.resize(x_fix, (X.shape[0], 6))
        Xs = np.hstack((X, x_fix))
#         Xs = Xs [:,:3]
        Pc = self.estimator.predict(Xs).clip(*self.LimitPcut)
        # debug begin
        #~ print(Xs.shape, Xs)
        #~ print('Pc', Pc.shape)
        #~ for p in Pc:
            #~ print('\t%.3f'%p)
        # debug end
        return Pc[0]
    
    def f_objectives(self, Lair=15, Lcut=100, Pst=945, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径.
        Stage(0)精加工, Stage(1+)粗加工.
        """
        # debug info
        info = {}
        # debug end
        
        t_st = t_st if self.Stage==0 else 0
        
        Dt = self.Dt
        
        def t(D, L, Vc, f):return 60* 3.14*D*L/(1000.*Vc*f)
        
        t_air = t(Dt, Lair, self.tVc, self.tF)
        t_cut = t(Dt, Lcut, self.tVc, self.tF)
        
        def Est():
            return Pst*(t_st + t_air + t_cut)
        
        def Eu():
#             def pu(Vc,D):return -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
#             def pu(Vc,D):return 1.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
#             def pu(Vc,D):return 0.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
            def pu(Vc,D):return 3.079*(1e3*Vc/(PI*D)) + 0.013*(1e3*Vc/(PI*D))**2 + 92.9
            Pu_t = pu(self.tVc, Dt)
            return Pu_t*t_air + Pu_t*t_cut
        
        def Emr():
            Pmr = self.Pcut()
            # debug begin
            info.update(Pmr=Pmr)
            # debug end
            return Pmr*t_cut
        
        def Eauc(Pcf=80, Phe=1000):
            return (Pcf+Phe)*(t_air + t_cut)
        
        #~ def T(Vc, f, Ap):return 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.33*Ap**0.28)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.03*Ap**0.31)
#         def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**0.95*Ap**0.33)
        def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.4*f**0.95*Ap**0.33)
        
        t_ect = t_cut/T(self.tVc, self.tF, self.tAp)
        
        def Ect():
            """ 只有精加工之后发生 """
            #~ return Pst*t_pct*t_ect *(1 if self.Stage==0 else 0)
            #~ return (Pst*t_pct+5340185.)*t_ect *(1 if self.Stage==0 else 0)
            return (Pst*t_pct+5340185.)*t_ect
#             return (Pst*t_pct+0.)*t_ect
        
        # debug begin
        e_st = Est()
        e_u = Eu()
        e_mr = Emr()
        e_auc = Eauc()
        e_ct = Ect()
        E = e_st+e_u+e_mr+e_auc+e_ct
        
        SEC = E / (0.785*(self.D**2-self.Dfinal**2)*Lcut)
#         SEC = (e_st+e_u+e_mr+e_auc+e_ct)/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        # debug end
        #~ SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + t_air + t_cut + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5, N=4):
            kt = Ch/400 + Ci/(0.75*ne)
            #~ return k0*Tp + ke*Tp + kt*t_ect
#             return k0*Tp/60 + ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6 + kt*t_ect/N
            cp_p1, cp_p2, cp_p3 = k0*Tp/60, ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6, kt*t_ect/N
            info.update(cp_p1=cp_p1,cp_p2=cp_p2,cp_p3=cp_p3)
            return cp_p1 + cp_p2 + cp_p3
        
        # debug begin
        info.update(Dt=Dt, t_st=t_st, t_air=t_air, t_cut=t_cut, t_pct=t_pct, t_ect=t_ect)
        info.update(e_st=e_st, e_u=e_u, e_mr=e_mr, e_auc=e_auc, e_ct=e_ct)
        info.update(E=E, SEC=SEC, Tp=Tp)
        info.update(D=self.D, Dfinal=self.Dfinal, Lcut=Lcut)
        # debug end
        
        return SEC, Tp, Cp(N=3), self.penaltyRa(), info
    
    def f_objectives_1(self, Lair=15, Lcut=100, Pst=335, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径.
        Stage(0)精加工, Stage(1+)粗加工.
        """
        # debug info
        info = {}
        # debug end
        
        t_st = t_st if self.Stage==0 else 0
        
        Dt = self.Dt
        
        def t(D, L, Vc, f):return 60* 3.14*D*L/(1000.*Vc*f)
        
        t_air = t(Dt, Lair, self.tVc, self.tF)
        t_cut = t(Dt, Lcut, self.tVc, self.tF)
        
        def Est():
            return Pst*(t_st + t_air + t_cut)
        
        def Eu():
#             def pu(Vc,D):return -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
#             def pu(Vc,D):return 1.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
#             def pu(Vc,D):return 0.8597*(1e3*Vc/(PI*D)) + 0.003*(1e3*Vc/(PI*D))**2 + 362.9
#             def pu(Vc,D):return 3.079*(1e3*Vc/(PI*D)) + 0.013*(1e3*Vc/(PI*D))**2 + 92.9
            def pu(Vc,D):return 1.029*(1e3*Vc/(PI*D)) + 0.013*(1e3*Vc/(PI*D))**2 + 29.95
            Pu_t = pu(self.tVc, Dt)
            return Pu_t*t_air + Pu_t*t_cut
        
        def Emr():
            Pmr = self.Pcut()
            # debug begin
            info.update(Pmr=Pmr)
            # debug end
            return Pmr*t_cut
        
#         def Eauc(Pcf=80, Phe=1000):
        def Eauc(Pcf=132, Phe=0):
            return (Pcf+Phe)*(t_air + t_cut)
        
        #~ def T(Vc, f, Ap):return 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.33*Ap**0.28)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.03*Ap**0.31)
#         def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**0.95*Ap**0.33)
#         def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.4*f**0.95*Ap**0.33)
        def T(Vc, f, Ap):return 60* 4.44e12/(Vc**6.568*f**1.278*Ap**0.20)
        
        t_ect = t_cut/T(self.tVc, self.tF, self.tAp)
        
        def Ect():
            """ 只有精加工之后发生 """
            #~ return Pst*t_pct*t_ect *(1 if self.Stage==0 else 0)
            #~ return (Pst*t_pct+5340185.)*t_ect *(1 if self.Stage==0 else 0)
            return (Pst*t_pct+5340185.)*t_ect
#             return (Pst*t_pct+0.)*t_ect
        
        # debug begin
        e_st = Est()
        e_u = Eu()
        e_mr = Emr()
        e_auc = Eauc()
        e_ct = Ect()
        E = e_st+e_u+e_mr+e_auc+e_ct
        
        SEC = E / (0.785*(self.D**2-self.Dfinal**2)*Lcut)
#         SEC = (e_st+e_u+e_mr+e_auc+e_ct)/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        # debug end
        #~ SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + t_air + t_cut + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5, N=4):
            kt = Ch/400 + Ci/(0.75*ne)
            #~ return k0*Tp + ke*Tp + kt*t_ect
#             return k0*Tp/60 + ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6 + kt*t_ect/N
            cp_p1, cp_p2, cp_p3 = k0*Tp/60, ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6, kt*t_ect/N
            info.update(cp_p1=cp_p1,cp_p2=cp_p2,cp_p3=cp_p3)
            return cp_p1 + cp_p2 + cp_p3
        
        # debug begin
        info.update(Dt=Dt, t_st=t_st, t_air=t_air, t_cut=t_cut, t_pct=t_pct, t_ect=t_ect)
        info.update(e_st=e_st, e_u=e_u, e_mr=e_mr, e_auc=e_auc, e_ct=e_ct)
        info.update(E=E, SEC=SEC, Tp=Tp)
        # debug end
        
        return SEC, Tp, Cp(N=3), self.penaltyRa(), info
    
    def penaltyRa(self, k=0, u=1, a=1e6, Ramax=1.6, Kmax=10):
        if self.Stage==-1:
            return 0
        g = self.tF**2/(8*0.8) - Ramax
        p = a**min(k, Kmax) *u* max(0, g)**2
        return p
    
    def Pseudo_Huber_loss(self, x, k=1):
        return k**2 * (math.sqrt(1+(x/k)**2)-1)
    
    def penaltyAp(self, ap_x, a=1e6, k=2):
        """ default a=10 """
        if self.Stage==-1:
            Limiter = self.LimitApf
#             ap_x = self.Ap_final
        else:
            Limiter = self.LimitApr
        gmin = Limiter[0] - ap_x    # min < ap_r
        gmax = ap_x - Limiter[1]    # ap_r < max
#         pmin = max(0, gmin)**k
#         pmax = max(0, gmax)**k
        pmin = max(0, gmin)
        pmax = max(0, gmax)
#         if pmin<1:
#             pmin = pmin**0.5
#         else:
#             pmin = pmin**2
#         if pmax<1:
#             pmax = pmax**0.5
#         else:
#             pmax = pmax**2
#         pmin = self.Pseudo_Huber_loss(pmin)
#         pmax = self.Pseudo_Huber_loss(pmax)
#         return a*(pmin + pmax)
        cis = np.array([pmin, pmax])
#         return a*cis.sum(), cis[cis>0].size
#         return a*cis.sum(), cis[cis>0].size, cis
        return a*np.power(cis,k).sum(), cis[cis>0].size, cis
#         return a*(pmin + pmax), cis[cis>0].size, cis
    
    def penaltyApf(self, ap_x, a=1e6, k=2):
        """ 0.5<=Kt_next<=1 """
        if self.Stage==-1:
            return 0, 0
        p_min_r = 0
        Kt_next = self.Kt - ap_x
        if Kt_next<=self.LimitApf[1]:
            le_min_r = self.LimitApf[0] - Kt_next
            p_min_r = max(0, le_min_r)
        return a*p_min_r**k, (1 if p_min_r>1e-6 else 0)
    
#     def penalty_3(self, a=1e6, t=2, k1=1, k2=2.5, k3=1):
    def penalty_3(self, a=1e6, t=2, k1=2.5, k2=1.5, k3=1):
        """ fVc > k1 * rVc   k1 = 1.
        rF  > k2 * fF    k2 = 2.5
        rAp > k3 * fAp   k3 = 1.
        """
        if self.Stage!=-1:
            return 0, 0, np.zeros(1)
        g_fVc = k1 * self.rVc_max - self.tVc
        g_rF = k2 * self.tF - self.rF_min
        g_rAp = k3 * self.tAp - self.rAp_min
#         ci1 = max(0, g_fVc)**t
#         ci2 = max(0, g_rF)**t
#         ci3 = max(0, g_rAp)**t
        ci1 = max(0, g_fVc)
        ci2 = max(0, g_rF)
        ci3 = max(0, g_rAp)
#         if ci2<1:
#             ci2 = ci2**0.5
#         else:
#             ci2 = ci2**2
#         ci1 = self.Pseudo_Huber_loss(ci1)
#         ci2 = self.Pseudo_Huber_loss(ci2)
#         ci3 = self.Pseudo_Huber_loss(ci3)
#         return a*(ci1+ci2+ci3)
#         cis = np.array([ci2, ci3])
#         cis = np.array([ci1, ])
        cis = np.array([ci2, ])
#         return a*cis.sum(), cis[cis>0].size
#         return a*cis.sum(), cis[cis>0].size, cis
        return a*np.power(cis,t).sum(), cis[cis>0].size, cis
#         return a*(ci2), cis[cis>0].size, cis
    
    def reward(self, mo=0, ea=0):
        f0, f1, f2, p0, info = self.f_objectives()
        mobj = np.array([f0, f1, f2])
        if ea:
            p_ap, np_ap, ci_ap = self.penaltyAp(self.tAp, a=1e7)
            p_3, np_3, ci_3 = self.penalty_3(a=1e7)
        else:
#             p_ap, np_ap, ci_ap = self.penaltyAp(self.tAp, a=3e3)
            p_ap, np_ap, ci_ap = self.penaltyAp(self.tAp, a=10)
#             p_3, np_3, ci_3 = self.penalty_3(a=3e3)
#             p_3, np_3, ci_3 = self.penalty_3(a=10)
            p_3, np_3, ci_3 = self.penalty_3(a=1)
        self.step_penalty = np.concatenate((ci_ap, ci_3))
#         self.step_penalty = ci_ap.copy()
#         mobj = np.tanh(mobj)
        # store solution begin
#         self.ep_objs += mobj
        # store solution end
        # keep step info begin
        self.step_objs = mobj.copy()
        # keep step info end
        # punish begin
        punish = 0.
        punish += np_ap
#         punish += np_ap + np_3
#         punish += p_ap
#         punish += p_3
        # penalty ap_f begin
        p_apf, np_apf = self.penaltyApf(self.tAp, a=100)
        if p_apf>0 and 0:
            punish += p_apf
            info.update(apf_done=1)
        # penalty ap_f end
        self.step_punish = punish
        self.ep_punish += punish
#         mobj += punish                  # pretrain
        # punish end
        # keep step info begin
        self.ep_objs += mobj
        # keep step info end
        done = 0
        if mo:
            # multiobjective R
#             if 0:
            if 1:
                pap, pf = p_ap, p_3
                (pap,),(pap_m,),(pap_a,), = self.zf_penalty([p_ap])
                (pf,),(pf_m,),(pf_a,), = self.zf_penalty1([p_3])
                punishx =  pap + pf
                R = -punishx
                R = -np.tanh(punishx)
                R = 0
            else:
                R = -np.tanh(punish)
        else:
            # single-objective
#             f0 *= 0.1
#             f1 *= 0.01
            fx, pap, pf = [f0,f1,f2][self.single_obj_type], p_ap, p_3
            self.ep_R += fx
#             punishx = 0
#             punishx =  np.sign(pap) + np.sign(pf) + np.sign(p_apf)
#             punishx =  np_ap*2 + np_3 + np_apf*0.01
#             punishx =  np_ap*2 + np_3*0.01  # ok
#             punishx =  np_ap*2 + np_3*0.1
#             punishx =  np_ap + np_3
#             punishx =  np_ap
            punishx = punish
            if punishx>0:
                R = -punishx * 30   # ok
#                 R = -punishx * 10
#                 R = -punishx * 5
                done = 2
            else:
#                 R = 0.5/fx
#                 R = 1.7/fx      # ok
                R = 2/fx
#                 R = 20/self.ep_R
                R = 0   # Moo !
#                 R = 3 - fx      # ok
#                 R = 5 - fx      # ok
#             R = -np_ap
#             R = - pap - pf - p_apf + 1/fx
#             R = -punishx + 1.7/fx
        return R, mobj, info, done
    
    def penalty_ap(self, ap_x):
        """ g(x)>=0 """
        if self.Stage==-1:
            Limiter = self.LimitApf
        else:
            Limiter = self.LimitApr
        ge_min = ap_x - Limiter[0]    # min < ap_r
        ge_max = Limiter[1] - ap_x    # ap_r < max
        return np.array([ge_min, ge_max])
        
    def penalty_rf(self, k1=2.5, k2=1.5, k3=1):
        """ g(x)>=0
        fVc > k1 * rVc   k1 = 1.
        rF  > k2 * fF    k2 = 2.5
        rAp > k3 * fAp   k3 = 1.
        """
        if self.Stage!=-1:
            return np.zeros(1)
        ge_fVc = self.tVc - k1 * self.rVc_max
        ge_rF = self.rF_min - k2 * self.tF
        ge_rAp = self.rAp_min - k3 * self.tAp
        return np.array([ge_rF])
    
    def init_phr(self):
        self.phr_eps = 1e-4
        self.phr_alpha = 1.5              # a>1
        self.phr_beta = 0.8        # (0,1)
        self.phr_sigma0 = 0.8
        self.phr_sigma = self.phr_sigma0
        self.phr_w0 = np.ones(3)
        self.phr_w = self.phr_w0.copy()
        self.phr_term_old = None
    
    def PHR(self, mo=0, ea=0):
        done = 0
        
        f0, f1, f2, p0, info = self.f_objectives()
        
        mobj = np.array([f0, f1, f2])
        self.step_objs = mobj.copy()
        
        fx = mobj[self.single_obj_type]
        # punish begin
        ci_ap = self.penalty_ap(self.tAp)
        ci_rf = self.penalty_rf()
        
        ci = np.concatenate((ci_ap, ci_rf))
        
        gi = np.fmax(0, self.phr_w-self.phr_sigma*ci)**2 - self.phr_w**2
        
        M = fx + 0.5*np.sum(gi)/self.phr_sigma
        
        R = -math.tanh(M)
        # debug begin
#         print('M', M, 'sigma', self.phr_sigma)
#         R = -math.tanh(fx)
        # debug end
        
        def g_norm():
            m = np.fmin(ci, self.phr_w/self.phr_sigma)**2
            return LA.norm(m)
        
        phr_term = g_norm()
        
        if phr_term<self.phr_eps:
            done = 1
#             return R, done, mobj, info
        else:
        
            if self.phr_term_old is not None and phr_term/self.phr_term_old>self.phr_beta:
                if self.phr_sigma<1e6:
                    self.phr_sigma *= self.phr_alpha
        
            self.phr_w = np.fmax(0, self.phr_w-self.phr_sigma*ci)
        self.phr_term_old = phr_term
        
        self.step_punish = phr_term
        self.ep_punish += phr_term
        mobj += phr_term                  # pretrain
        # punish end
        # keep step info begin
#         Rs = self.step_objs - mobj
#         self.step_objs = mobj.copy()
        self.ep_objs += mobj
        # keep step info end
        if mo:
            # multiobjective R
            R = -np.tanh(phr_term)
        
        return R, done, mobj, info
    
    def reward_dp(self, mo=0, ea=0):
        f0, f1, f2, p0, info = self.f_objectives()
        
        mobj = np.array([f0, f1, f2])
        self.step_objs = mobj.copy()
        
        fx = mobj[self.single_obj_type]
        # punish begin
#         ci_ap = self.penalty_ap(self.tAp)
#         ci_rf = self.penalty_rf()
#         ci = np.concatenate((ci_ap, ci_rf))
        punish = 0
        
        if self.StageFin and self.tAp<self.Kt:
            punish = 1
        
        self.step_punish = punish
        self.ep_punish += punish
        mobj += punish                  # pretrain
        # punish end
        # keep step info begin
#         Rs = self.step_objs - mobj
#         self.step_objs = mobj.copy()
        self.ep_objs += mobj
        # keep step info end
        if mo:
            R = -np.tanh(punish)
        else:
            R = -math.tanh(f0*0.1) - punish
        
        return R, mobj, info
    
    def reward_hv(self, reward=0):
        # shared pop begin
        if self.ep_count>1000 and self.ep_count%100==0:
            self.update_moo_pop()
        # shared pop end
        if self.pop is None:
            self.pop = np.array( [self.ep_objs[:2]] )
#             pass
        else:
            test_p = np.vstack( (self.pop, self.ep_objs[:2]) )
            hv = pg.hypervolume(test_p)
            reward = hv.exclusive(len(self.pop), hv.refpoint(offset=0.1))
            if reward>0:
#                 indices = pg.select_best_N_mo(test_p, 100)
#                 self.pop = test_p[indices]
                reward = math.log(reward+1)
#                 reward = math.tanh(reward)
        return reward
    
    def _action_scale3(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        Limiter = self.Limit3
        k = (Limiter[1] - Limiter[0])/ 2.
        b = (Limiter[1] + Limiter[0])/ 2.
        return k * action + b
    
    # TODO: fix range
    def _action_scale3_dp(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.StageFin:
            Limiter = self.Limit3f.copy()
            Limiter[1][1] = min(Limiter[1][1], self.rF_min/1.5)
        else:
            Limiter = self.Limit3r.copy()
        Limiter[1][2] = min(Limiter[1][2], self.Kt)
        k = (Limiter[1] - Limiter[0])/ 2.
        b = (Limiter[1] + Limiter[0])/ 2.
        return k * action + b
    
    def begin_step_dp(self, action, eps=0.05):
        self.tVc, self.tF, self.tAp = self._action_scale3_dp(action)
        # debug begin
        print(self.Dt, self.Kt, self.tVc, self.tF, self.tAp)
        # debug end
        def get_states():
            return [self.tVc, self.tF, self.tAp, self.Dt, self.Kt, self.Kt-self.tAp]
        assert self.Kt>0
        # Generate State Params Begin
        finish = 0
        if self.StageFin:
            finish = 1
            self.Stage = -1
        if not finish:
            self.rVc_max = max(self.rVc_max, self.tVc)
            self.rF_min = min(self.rF_min, self.tF)
            self.rAp_min = min(self.rAp_min, self.tAp)
        # store solutions begin
        self.ep_solution['sol'].append( get_states() )
        # store solutions end
        # Generate State Params End
        return finish
    
    def begin_step(self, action, eps=0.05, no_scale=False):
        if no_scale:
            self.tVc, self.tF, self.tAp = action
        else:
            self.tVc, self.tF, self.tAp = self._action_scale3(action)
        # debug begin
#         if self.debug:
#             print('action', action)
#             print('VcFap', self.tVc, self.tF, self.tAp)
        # debug end
        def get_states():
            return [self.tVc, self.tF, self.tAp, self.Dt, self.Kt, self.Kt-self.tAp]
        assert self.Kt>0
        # Generate State Params Begin
        finish = 0
#         if self.tAp>=self.Kt or (self.Kt-self.tAp)/self.tAp<eps:
        if self.tAp>=self.Kt:
#             if self.Kt>self.LimitApf[1] and 1:
            if self.Kt>self.LimitApf[1] and 0:
                self.tAp = self.LimitApf[1]
            else:
                self.Ap_final = self.tAp
                self.tAp = self.Kt
                # F constraint begin
#                 self.tF = min(self.rF_min/1.5, self.tF)
                # F constraint end
                finish = 1
                self.Stage = -1
        if not finish:
            self.rVc_max = max(self.rVc_max, self.tVc)
            self.rF_min = min(self.rF_min, self.tF)
            self.rAp_min = min(self.rAp_min, self.tAp)
        # store solutions begin
        self.ep_solution['sol'].append( get_states() )
        # store solutions end
        # Generate State Params End
        return finish
    
    def end_step(self):
        self.Dt -= self.tAp*2
        self.Kt -= self.tAp
        self.Stage += 1
        if self.Kt<=self.LimitApf[1]:
            self.StageFin = 1
    
    def reset(self):
        self.init_states()
        return self.make_obs()
    
    def step(self, action):
        """ action: [Vc,f,Ap] """
        info, done = {}, 0
        action = np.float32(action)
#         done = self.begin_step_dp(action)
#         reward, _, step_info = self.reward_dp(mo=self.moo)
        done = self.begin_step(action)
        reward, _, step_info, done1 = self.reward(mo=self.moo)
        done = done1 or done
        if step_info.get('apf_done', 0):
            done = 2
#         reward, term, _, step_info = self.PHR(mo=self.moo)
#         done += term
        self.end_step()
        
        self.StageId += 1
        
        obs = self.make_obs()
        
        # debug begin
        if 'step_obj' in step_info:
            info.update(step_obj=step_info['step_obj'])
        # debug end
        
        if self.moo:
            info.update(pref=self.ref_point)
        # store solutions begin
        if done:
            if done==1:
#                 reward = 1000/self.ep_R
                if self.ep_punish<1e-6 and 1:
                    best_obj = self.best_solution.get('obj', None)
                    if best_obj is None or best_obj[self.single_obj_type]>self.ep_objs[self.single_obj_type]:
                        self.best_solution.update(obj=self.ep_objs.copy(), 
                                                  sol=self.get_solution(self.ep_objs),
                                                  R=self.ep_R)
                        info.update(solution=[self.get_solution(self.ep_objs), self.ep_objs[self.single_obj_type], 1])
                    info.update(f0=self.ep_objs[self.single_obj_type])
                    info.update(penalty=self.ep_punish)
#                     info.update(mobj=self.ep_objs.copy()[:2])
#                     info.update(mobj=np.clip(self.ep_objs.copy()[:2], None, [40, 1000]))
                    info.update(mobj=np.clip(self.ep_objs.copy()[:2], None, self.ref_clip))
                    if self.id==0:
                        info.update(ref_p=self.ref_point)
                    info.update(sol='n_cuts: %d'%len(self.ep_solution['sol']))
                    info.update(moo_x=self.ep_solution['sol'])
                elif 0:
                    info.update(f0=self.ep_objs[self.single_obj_type])
                    info.update(penalty=self.ep_punish)
                    info.update(mobj=self.ep_objs.copy()[:2])
                    info.update(sol='n_cuts: %d'%len(self.ep_solution['sol']))
            elif done==2:
                info.update(mobj_ill=np.array([0.,0.]))
            
            if self.debug and 10:
                print('EP', self.ep_count)
                print('done', self.get_solution(self.ep_objs))
                print(f'reward: {self.ep_R:.3f}')
#                     print('best obj', self.best_solution.get('obj'))
                print('best sol', self.best_solution.get('sol'))
                print("reward: {:.3f}".format(self.best_solution.get('R', -999)))
        # store solutions end
        
        return obs, float(reward), bool(done), info
    
    def penaltySumAp(self, sum_ap, n_err, a=1e6, k=2):
        h = abs(sum_ap-self.K0)
        #~ return h**k
        #~ return h**k + 2*abs(n_err)**k
        #~ return h**k + 50*abs(n_err)**2
        return a*h**k + a*abs(n_err)**k
    
    def function(self, X, mo=0, penalty=1, eval=0, no_scale=False):
        """ For NiaPy """
        self.init_states()
        
        # debug info
        dbg_info = []
        # debug end
        X = np.array(X).reshape(-1, 3)
        mobj_ep = np.zeros(3)
        n_x = X.shape[0]
        n_sol = 0
        R = 0
        for x in X:
            done = self.begin_step(x, no_scale=no_scale)
            r_, mobj, step_info, _ = self.reward(mo=mo, ea=1)
            # debug info
#             step_info.update(done=done)
            dbg_info.append(step_info)
            # debug end
            self.end_step()
            R += r_
            mobj_ep += mobj
            n_sol += 1
            if done:
                break
        
        if penalty:
            # penalty not done begin
            self.Stage = -1
            p_apf, np_apf, _ = self.penaltyAp(self.tAp)
            # penalty not done end
            sum_ap = self.get_solution(self.ep_objs, getAp=1)
            penaltly_aps = self.penaltySumAp(sum_ap, n_err=n_sol-n_x)
            mobj_ep += penaltly_aps
            mobj_ep += p_apf
            self.ep_punish += penaltly_aps + p_apf
            punishx = np_apf + np.sign(penaltly_aps)
            if punishx>0:
                R = -punishx if R>0 else R-punishx
        
        if eval:
            dbg_info.append([self.ep_punish, penaltly_aps])
            return self.get_solution(self.ep_objs), dbg_info
        
        if mo:
            return mobj_ep
        return R
    
    def function_plot(self, x, stage=None):
        self.init_states()
        
        if stage is not None:
            self.Stage = stage
        
        done = self.begin_step(x)
        
        f0, f1, f2, p0, info = self.f_objectives()
        f0 = info['E']
        self.end_step()
        
        return f0, f1, f2, self.tVc, self.tF, self.tAp, info
    
    def get_solution(self, score, getAp=0):
        sol = '----- SEC %.3f, Tp %.3f, Cp %.3f -----\n'%(score[0], score[1], score[2])
        last = len(self.ep_solution['sol'])
        cut_depth = 0
        for i, cut in enumerate(self.ep_solution['sol'], 1):
            cut_depth += cut[2]
            i = 'F' if i==last else str(i)
            sol += '[%s] Vc:%.3f f:%.3f Ap:%.3f Dt:%.3f Kt:%.3f Kt1:%.3f\n'%(i, *cut)
        sol += '------------------ %.6fmm (K0 %.6f)------------------\n'%(cut_depth, self.K0)
#         sol += 'OBJECTIVE: %.3f, Penalty %.3f'%(self.ep_objs[self.single_obj_type], self.ep_punish)
        sol += 'Penalty %.3f'%(self.ep_punish, )
        if getAp:
            return cut_depth
        return sol

