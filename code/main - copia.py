
"""
Created on Tue Jan 12 10:19:35 2021

@author: HOLGER
"""



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import cm
#plt.rc('text', usetex=False)

from modelado_elastico2d import elastic_modeling_conv_cpu
from funciones import normalize_2d, graph, interpol


#%% CARGAR DATOS

Vp=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\vp.npy')
Vs=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\vs.npy')
Rho=np.load(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\half_space\rho.npy')

#Vp=np.loadtxt(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos SEAM\modelo_1\vp_modelo1.txt')
#Vs=np.loadtxt(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos SEAM\modelo_1\vs_modelo1.txt')
#Rho=np.loadtxt(r'C:\Users\USUARIO\Documents\Python Scripts\proyecto\python\modelado\modelos SEAM\modelo_1\rho_modelo1.txt')

(Nz,Nx)=Vp.shape

'''
#Capa de vacio
vacio=np.zeros((10,Nx))
Vp=np.concatenate((vacio,Vp),axis=0)
Vs=np.concatenate((vacio,Vs),axis=0)
Rho=np.concatenate((vacio,Rho),axis=0)

'''
#Debemos interpolar los modelos para dejarlos de 1m de separación
#[Vp,Vs,Rho]=[interpol(Vp),interpol(Vs),interpol(Rho)]


dh=1                     #Space sampling
dx=dh; dz=dh             #Horizontal and vertical step
x=np.arange(0,dh*Nx,dh)   #Horizontal axis
z=np.arange(0,dh*Nz,dh)   #Vertical axis

#%% TOPOGRAFÍA
'''
#Ahora hagamos que la capa de vacio sea de 50m
temp=np.nonzero(Vp[:,0])
topo=np.min(temp)
for i in np.arange(1,Nx):
    temp=np.nonzero(Vp[:,i])
    k2=np.min(temp)
    if k2<topo:
        topo=k2
        
Vp=Vp[topo-50:,:]
Vs=Vs[topo-50:,:]
Rho=Rho[topo-50:,:]
'''

top=np.zeros([Rho.shape[1],1])
for i in np.arange(Rho.shape[1]):
    id_top=np.array([np.where(Rho[:,i]==0)])
    if id_top.size>0:
        id_top=np.max(id_top)
        top[i]=id_top*dz

#%% PARÁMETROS DE MODELADO

spx=500; spz=10;                  #Source position
gz=0; gx=np.arange(0,1000,6)     #Receivers position
Ts=2e-3                            #Time sampling(No cambiar)
#dt=0.99/(Vmax*1.1667*np.sqrt(1/(dx^2)+1/(dz^2)));
dt=1e-4 
tf=1
Ts_field=0.05

#%% FUENTE WAVELET
fq=10                         #Central frequency
t=np.arange(0,tf+dt,dt)          #Time axis
t0=1/fq                      #Delay source
source=(1-2*(np.pi*fq*(t-t0))*(np.pi*fq*(t-t0)))*np.exp(-(np.pi*fq*(t-t0))*(np.pi*fq*(t-t0)))  #Source wavelet
     
    
#%%PARÁMETROS DE ELASTICIDAD        
    
Mu=(Vs*Vs)*Rho
Lambda=(Vp*Vp)*(Rho)-2*Mu
Vmax=np.max(Vp)

#%%LLAMAR A LA FUNCION
(gather_z,gather_x,field_vz,field_vx,field_szz,field_sxx,field_sxz)=elastic_modeling_conv_cpu(Mu,Lambda,Rho,dz,dx,Ts,spz,spx,gz,gx,source,dt,fq,Vmax,Ts_field)

#%% FIGURES
plt.figure(1)
plt.ion()
for j in np.arange(0,20):
    plt.imshow(field_vz[:,:,j],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[x[0],x[x.shape[0]-1],z[z.shape[0]-1],z[0]],cmap=cm.jet)
   # plt.axhline(y=50, color="black", linestyle="-")#Línea topografía
    #plt.axhline(y=250, color="black", linestyle="--")
  #  plt.plot(x,65+IGS ,color='darkred',linestyle='--')
    plt.plot(spx,spz, marker='*', color='black')
    plt.plot(x,top, color='black',lw=0.4) 
    #plt.plot(x,top+3, color='black',lw=0.4)
    #plt.plot(x,top+15, color='black',lw=0.4)
    #plt.plot(x,top, color='black',lw=0.4)
    #plt.text(1, 13, 'zona random_media!', fontsize=10, color='black')
   # plt.plot(720,65, marker='.', color='black')
    plt.xlabel(r'Distance (m)')
    plt.ylabel(r'Depth (m)')
    plt.title(r'Time 3 (s)')
    plt.draw()
    plt.pause(0.4)
    ax = plt.gca()

'''
    rect = patches.Rectangle((326,26),
                630,
                 8,
                 linewidth=2,
                 edgecolor='black',
                 fill = False)

    ax.add_patch(rect)
 '''   
    
plt.ioff()
plt.show()


#%%
plt.figure(2)
plt.subplot(121)
plt.imshow(gather_x[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8, extent=[gx[0],gx[gx.shape[0]-1],t[t.shape[0]-1],t[0]],cmap=cm.gray)
plt.axis([gx[0],gx[gx.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Horizontal shot gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

plt.subplot(122)
plt.imshow(gather_z[:,:],aspect='auto',vmin=-1e-8,vmax=1e-8,extent=[gx[0],gx[gx.shape[0]-1],t[t.shape[0]-1],t[0]], cmap=cm.gray)
plt.axis([gx[0],gx[gx.shape[0]-1],t[t.shape[0]-1],t[0]])
plt.title('Vertical shot gather')
plt.ylabel(r'Time (s)')
plt.xlabel(r'Distance (m)')
plt.colorbar(orientation='horizontal')

#plt.text(3, 11, u'Vacío', fontsize = 10, horizontalalignment='center', verticalalignment='center',bbox ={'facecolor':'white','pad':5}) Texto en imagenes
plt.show()

'''grafica de una seccion
plt.imshow(prueba,extent=[0,14,0,14],cmap='binary')
plt.ylabel(r'Profundidad [m]')
plt.xlabel(r'Distancia [m]')
x_ticks=np.arange(0, 14, 1)
plt.xticks(x_ticks)
plt.yticks(x_ticks)
plt.title(r'Modelo de Densidad [Kg/m^3]')
plt.grid(which="major",color='0.65',linestyle='dotted')
plt.text(3, 12, u'Vacío', fontsize = 10, horizontalalignment='center', verticalalignment='center',bbox ={'facecolor':'white','pad':5})
plt.text(11, 3, u'Tierra', fontsize = 10, horizontalalignment='center', verticalalignment='center',bbox ={'facecolor':'white','pad':5})
plt.text(3, 7, u'Capa ficticia', fontsize = 10, horizontalalignment='center', verticalalignment='center',bbox ={'facecolor':'white','pad':5})
plt.grid(True)
plt.show()
'''

#np.save('gather_x',gather_x)
#np.save('gather_z',gather_z)
#np.save('simulacion',field_vz)
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\fuente_arriba\f=50\a=1\e=0.5\gather_z.npy',gather_z)  
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\fuente_arriba\f=50\a=1\e=0.5\gather_x.npy',gather_x)  
np.save(r'C:\Users\USUARIO\Documents\Python Scripts\primer articulo\fuente_arriba\f=50\a=1\e=0.5\simulacion.npy',field_vz) 













'''para graficar cpml-zona
vs=np.delete(Vs,np.arange(6),axis=0)
sec=vs[:,0:60]

vs=np.concatenate((vs,sec),axis=1)
sec=vs[47:50,:]
vs=np.concatenate((vs,sec),axis=0)

plt.figure(1)    
plt.imshow(vs,aspect='auto',extent=[x[0],x[x.shape[0]-1],z[z.shape[0]-1],z[0]])
plt.ylabel('Depth [m]')
plt.xlabel('Distance [m]')
plt.colorbar(label='[m/s]')
plt.axhline(y=100, xmin=0.043, xmax=.967,color='black',linestyle='dotted')
plt.axvline(x=90, ymin=0.05, ymax=50,color='black',linestyle='dotted')
plt.axvline(x=2040, ymin=0.05, ymax=50,color='black',linestyle='dotted')
ax = plt.gca()

rect = patches.Rectangle((0,0),
                 90,
                 120,
                 linewidth=0.3,
                 facecolor = 'tomato',
                 fill = True)

ax.add_patch(rect)
ax = plt.gca()

rect = patches.Rectangle((50,100),
                 2100,
                 10,
                 linewidth=0.3,
                 facecolor = 'tomato',
                 fill = True)

ax.add_patch(rect)

ax = plt.gca()

rect = patches.Rectangle((2040,00),
                 200,
                 120,
                 linewidth=0.3,
                 facecolor = 'tomato',
                 fill = True)

ax.add_patch(rect)
plt.title('Vs Model')
plt.show()
'''









