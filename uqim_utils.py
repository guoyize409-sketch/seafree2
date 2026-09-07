"""Python-3-compatible vendored copy of the public FUnIE-GAN UIQM code.

Source: https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py
The evaluation entry point applies the same integer block-count compatibility
to the upstream implementation when loading it.
"""
from scipy import ndimage
import numpy as np
import math

def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x); K = len(x)
    T_a_L = math.ceil(alpha_L*K); T_a_R = math.floor(alpha_R*K)
    return (1/(K-T_a_L-T_a_R))*sum(x[int(T_a_L+1):int(K-T_a_R)])

def s_a(x, mu):
    return sum(math.pow((pixel-mu), 2) for pixel in x)/len(x)

def _uicm(x):
    R=x[:,:,0].flatten(); G=x[:,:,1].flatten(); B=x[:,:,2].flatten()
    RG=R-G; YB=((R+G)/2)-B
    mu_a_RG=mu_a(RG); mu_a_YB=mu_a(YB)
    return (-0.0268*math.sqrt(mu_a_RG**2+mu_a_YB**2) +
            0.1586*math.sqrt(s_a(RG,mu_a_RG)+s_a(YB,mu_a_YB)))

def sobel(x):
    dx=ndimage.sobel(x,0); dy=ndimage.sobel(x,1)
    mag=np.hypot(dx,dy); mag*=255.0/np.max(mag)
    return mag

def eme(x, window_size):
    k1=int(x.shape[1]/window_size); k2=int(x.shape[0]/window_size)
    w=2./(k1*k2); x=x[:window_size*k2,:window_size*k1]; val=0
    for l in range(k1):
        for k in range(k2):
            block=x[k*window_size:window_size*(k+1),l*window_size:window_size*(l+1)]
            max_=np.max(block); min_=np.min(block)
            if min_!=0.0 and max_!=0.0: val+=math.log(max_/min_)
    return w*val

def _uism(x):
    R=x[:,:,0]; G=x[:,:,1]; B=x[:,:,2]
    r_eme=eme(np.multiply(sobel(R),R),10)
    g_eme=eme(np.multiply(sobel(G),G),10)
    b_eme=eme(np.multiply(sobel(B),B),10)
    return 0.299*r_eme+0.587*g_eme+0.144*b_eme

def _uiconm(x, window_size):
    k1=int(x.shape[1]/window_size); k2=int(x.shape[0]/window_size)
    w=-1./(k1*k2); x=x[:window_size*k2,:window_size*k1]; val=0
    for l in range(k1):
        for k in range(k2):
            block=x[k*window_size:window_size*(k+1),l*window_size:window_size*(l+1),:]
            max_=np.max(block); min_=np.min(block); top=max_-min_; bot=max_+min_
            if not math.isnan(top) and not math.isnan(bot) and bot!=0.0 and top!=0.0:
                val+=(top/bot)*math.log(top/bot)
    return w*val

def getUIQM(x):
    x=x.astype(np.float32)
    return 0.0282*_uicm(x)+0.2953*_uism(x)+3.5753*_uiconm(x,10)
