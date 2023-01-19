# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 20:20:03 2022

@author: HOLGER
"""
import numpy as np
import math
from scipy import special


def random_media_2d(*args):
    (Y,Z)=np.meshgrid(args[0]-np.mean(args[0]),args[1]-np.mean(args[1]))
    
    if len(args)<5:
        acf='gaussian'
    else:
        acf=args[4]
    
    r = np.sqrt(Y**2+Z**2);
    
    if acf=='von karman':
        R = np.fft.fftshift(((args[2]**2)*(2**(1-args[5]))/math.gamma(args[5]))*((r/args[3])**args[5])*special.kn(1,r/args[3]))
        R=np.nan_to_num(R)
    elif acf=='exponential':
        R = np.fft.fftshift((args[2]**2)*np.exp(-r/args[3]))
    elif acf=='gaussian':
        R = np.fft.fftshift((args[2]**2)*np.exp(-(r**2)/args[3]**2))
    
    # PSDF
    P=np.fft.fftn(R);
    
    #Random phase
    A=np.sqrt(np.abs(np.random.randn(P.shape[0],P.shape[1])));
    Af=np.fft.fftn(A)
    phi=np.angle(Af)
    
    #Random media
    Ny=args[0].size;
    Nz=args[1].size;
    k=np.sqrt(Ny*Nz);
    M=k*np.real(np.fft.ifftn(np.sqrt(P)*np.exp(1j*phi)));
    return(M)


def interpol(F,h,v):
    (Nz,Nx)=F.shape
    #Vp
    a=np.linspace(F[0],F[1],h)
    for i in np.arange(1,Nz-1):
      b=np.linspace(F[i],F[i+1],h)
      p=np.concatenate((a,b),axis=0)
      a=p
    F=np.transpose(a)
    a=np.linspace(F[0],F[1],v)
    for i in np.arange(1,Nx-1):
      b=np.linspace(F[i],F[i+1],v)
      p=np.concatenate((a,b),axis=0)
      a=p
    F=np.transpose(a)  
    return(F)