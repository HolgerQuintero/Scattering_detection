# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 09:42:46 2022

@author: holger
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#%%
Dx=1000; Dz=500;
dh=1;
dx=dh;  dz=dh;
x=np.arange(0,Dx+dx,dx); z=np.arange(0,Dz+dz,dz);
Nx=x.size; Nz=z.size;

mask=np.ones((Nz,Nx))

vp=mask*1500
vs=mask*600
rho=mask*2200


#%%
M=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\M.npy')

n=50
mascara=np.zeros(vs.shape)
for i in np.arange(50,450):
    mascara[np.int(0):np.int(0+n),i]=1
    

#%%Modelo final
vs=vs+mascara*np.abs(M)*vs 
#vp=vp+mascara*np.abs(M)*vp 
#rho=rho+mascara*np.abs(M)*rho 

#%%
mask4=np.zeros((Nz,Nx))
for i in np.arange(Nz):
    for j in np.arange(Nx):
        if np.sqrt(np.power(i-50,2)+np.power(j-200,2)) <= 20:
            mask4[i,j]=1


for i in np.arange(Nx):
   for j in  np.arange(Nz):
       if mask4[j,i]==1:
           vp[j,i]=2400
           vs[j,i]=1000
           rho[j,i]=2500

#%%Capa de vacio
vacio=np.zeros((10,Nx))
vp=np.concatenate((vacio,vp),axis=0)
vs=np.concatenate((vacio,vs),axis=0)
rho=np.concatenate((vacio,rho),axis=0)


#%%

plt.figure(1)    
plt.imshow(vs,aspect='auto',extent=[x[0],x[x.shape[0]-1],z[z.shape[0]-1],z[0]])
plt.ylabel('Depth [m]')
plt.xlabel('Distance [m]')
plt.colorbar(label='[m/s]')
plt.title('Vs model',style='italic')
plt.show()

#%% Salvar dato
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\vs',vs)  
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\vp',vp)  
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\rho',rho)  
    
