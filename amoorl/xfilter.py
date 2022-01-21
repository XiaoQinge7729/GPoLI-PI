import numpy as np


class RunningStat(object):
    def __init__(self, shape, a=1.0, b=1.0):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
        
        self.A = a
        self.B = b
    
    def reset(self):
        self._n = 0
        self._M.fill(0)
        self._S.fill(0)
    
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            #~ self._M[...] = self.A*oldM + (self.B*x - self.A*oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class XFilter:

    def __init__(self, shape, P=0.0, I=1.0, clip=10.0):
        self.clip = clip
        
        self.rs = RunningStat(shape, a=0.7, b=1.3)
        
        self.Kp = P
        self.Ki = I
        self.SetPoint = np.ones(shape)
        self.PTerm = np.zeros(shape)
        self.ITerm = np.zeros(shape)
        self.windup_guard = 1000.0
        
        self.reset_once = 1
        
    def pid(self, feedback_value):
        error = self.SetPoint - feedback_value
        
        self.PTerm = self.Kp * error
        self.ITerm += error * 1
        
        self.ITerm = self.ITerm.clip(-self.windup_guard, self.windup_guard)
        
        alpha = self.PTerm + self.Ki * self.ITerm
        
        return np.power( 2, alpha.clip(-30, 30) )
    
    def adaptive(self, feedback_mean):
        #~ alpha = 1/(feedback_mean+1e-8)
        alpha = 1/np.maximum(feedback_mean**2, 1e-8)
        
        #~ return alpha.clip(-1e6, 1e6)
        return alpha.clip(1e-8, 1e8)
    
    def __call__(self, x, update=True):
        #~ if self.reset_once and self.rs.n==3000:
            #~ self.rs.reset()
            #~ self.reset_once = 0
        if self.rs.n>0:
            feedback_mean = self.rs.mean
        else:
            feedback_mean = np.asarray(x)
        alpha = self.adaptive(feedback_mean)
        x = x * alpha
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        if update:
        #~ if update and x[0]>0:
            self.rs.push(x)
        #~ if self.clip:
            #~ x = np.clip(x, -self.clip, self.clip)
        return x, feedback_mean, alpha



