
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
#import torch                               #<- This line may or may not be necessary
from torch import pow, add, mul, div, sqrt

def lorentzActivation(w, w0, wp, g):
  #wp*wp*w*g/( (w0*w0 - w)*(w0*w0 - w) + w*w*g*g ), Complex component of lorentz activation function
  e2 = div(mul(mul(pow(wp,2),w),g), add(pow(add(pow(w0,2),-w),2), pow(mul(w,g),2)))
  return e2
  
class LAF(nn.Module):
  def __init__(self):
    super(LAF, self).__init__()
    
  def forward(self,w):
    return lorentzActivation(w,0,wp,g)  #assume w0 = 0, other values must also be predefined
    
  
