
"""
This is the module where the imaginary lorentz activation function is defined.
"""
# Own modules

# Built in
import math
# Libs
import numpy as np

# Pytorch module
import torch.nn as nn
import torch
from torch import pow, add, mul, div, sqrt

def lorentzActivation(w, w0, wp, g):
  #wp*wp*w*g/( (w0*w0 - w)*(w0*w0 - w) + w*w*g*g ), Complex component of lorentz activation function
  e2 = div(mul(mul(pow(wp,2),w),g), add(pow(add(pow(w0,2),-w),2), pow(mul(w,g),2)))
  return e2
  
  
class LAF(nn.Module):
  
"""
Lorentz Activation Function with preset, fixed parameters
"""

  def __init__(self):
    super(LAF, self).__init__()
    
  def forward(self,w):
    return lorentzActivation(w,0,wp,g)  #assume w0 = 0, other values must also be predefined

  
  
class LLAF(nn.Module):
  
"""
Learnable Lorentz Activation function without preset parameters, except w0 = 0
"""

  def __init__(self, g = None, wp = None):
    super(LLAF, self).__init__()
    
    # Save g and wp as tracked parameters of this function module
    if g == None:
      self.g = Parameter(torch.tensor(0.0))
    else:
      self.g = Paramter(torch.tensor(g))
      
    if wp == None:
      self.wp = Parameter(torch.tensor(0.0))
    else:
      self.wp = Paramter(torch.tensor(wp))
    
    self.g.requiresGrad = True
    self.wp.requiresGrad = True
    
  def forward(self,w):
    return lorentzActivation(w,0,self.wp,self.g)  #assume w0 = 0
    
  
