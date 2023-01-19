# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:37:25 2022

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
from random_media_2d import random_media_2d


#%%
dh=1;
Dy=1000; Dz=500;
dy=dh; dz=dh;
y=np.arange(0,Dy+dz,dy); z=np.arange(0,Dz+dz,dz);
epsilon=0.5;
a=10;

np.random.default_rng()
M=random_media_2d(y,z,epsilon,a,'exponential')

#%% Figura
plt.figure   
plt.imshow(M,cmap='jet',aspect='auto',extent=[y[0],y[y.shape[0]-1],z[z.shape[0]-1],z[0]])
plt.ylabel('Z')
plt.xlabel('X')
plt.colorbar()
plt.show()


#%% Salvar dato
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\M',M)  