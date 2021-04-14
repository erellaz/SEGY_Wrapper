'''
2D acoustic wave equation propagator, using Deepwave

Here we will:
    -define propagator parameters
    -define a shot geometry
    -Load a numpy array with the velocity model previously prepared from a SEGY
    -run the propagator
    -extract shots, resample along the time dimension
    -save the shots in compressed numpy array on disk
    -export the shots to SEGY
'''

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import deepwave
import SEGY_wrapper as wrap

#______________________________________________________________________________
#User parameters: 
    
# Propagator parameters
freq = 12 # source max frequency in Hz
dx = [12.5,12.5] # Float or list of floats containing cell spacing in each dimension ordered as [dz, dy, (dx)]
dt = 0.001 # Propagator time step in s
nt = int(5 / dt) # insert shot length in seconds 
num_dims = 2 #  2D or 3D

# Survey parameters
num_shots = 1 #10
num_sources_per_shot = 1
num_receivers_per_shot = 420
source_spacing = 800.0 # meters
receiver_spacing = 12.5 # meters

# Compute parameters, CPUs or GPUs
#device = torch.device('cuda:0') # GPU
device = torch.device("cpu") #CPU

#The compressed Numpy array with all the shots, resampled in time
time_decim=6 # decimation of the shots in the time direction before saving shots to disk

#______________________________________________________________________________
# Load a subset of a SEGY into a NUMPY array using the SEGY Wraper
model_true=wrap.Segy2Numpy('vel_z6.25m_x12.5m_exact.segy',subsetz=(None,None,4),subsety=(2000,4000,2))

#______________________________________________________________________________
# Print informations and make pictures for QC
ny = model_true.shape[1] # Number of samples along y
nz = model_true.shape[0] # Number of depth samples, ie nbr samples along z
print("Velocity model Information:")
print("Velocity model size, ny , nz:", ny,nz)
print("Velocity model size in meters, Y and Z:",(ny-1)*dx[1],(nz-1)*dx[0])
Vvmin, Vvmax = np.percentile(model_true, [0,100])
print("Velocity min and max:", Vvmin, Vvmax)
#plt.imshow(model_true, cmap=plt.cm.jet, vmin=Vvmin, vmax=Vvmax)
plt.imsave('velocity_model_for_prop.png',model_true,
           cmap=plt.cm.jet, vmin=Vvmax, vmax=Vvmax)
#Compute stability condition
dtmax=wrap.CourantCondition(dx,num_dims,Vvmax)
print("Grid size:",dx)
print("Time step, number of time samples", dt,nt)
print("Stability condition on the time step dt:",dt,"<",dtmax)

#______________________________________________________________________________
# Convert from NUMPY array to torch tensor
model_true = torch.Tensor(model_true) # Convert to a PyTorch Tensor

#______________________________________________________________________________
# Define survey Geometry
# Create arrays containing the source and receiver locations
# x_s: Source locations [num_shots, num_sources_per_shot, num_dimensions]
# x_r: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions]
x_s = torch.zeros(num_shots, num_sources_per_shot, num_dims)
#x_s[:, 0, 1] = torch.arange(num_shots).float() * source_spacing
x_s[:, 0, 1] = torch.ones(num_shots).float() * dx[1]*100 #position VSP along y

x_r = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
x_r[0, :, 1] = torch.arange(num_receivers_per_shot).float() * receiver_spacing
x_r[:, :, 1] = x_r[0, :, 1].repeat(num_shots, 1)

x_v = torch.zeros(num_shots, num_receivers_per_shot, num_dims)
x_v[0, :, 0] = torch.arange(num_receivers_per_shot).float() * receiver_spacing # Build depth range, Third index: 0=z, 1=y, (2=x if 3D)
x_v[0, :, 1] = torch.ones(num_receivers_per_shot).float() * dx[1]*100 #position VSP along y
x_v[:, :, 0] = x_v[0, :, 0].repeat(num_shots, 1) #build one for each shot
#______________________________________________________________________________
# Create true source amplitudes [nt, num_shots, num_sources_per_shot]
# I use Deepwave's Ricker wavelet function. The result is a normal Tensor - you
# can use whatever Tensor you want as the source amplitude.
source_amplitudes_true = (deepwave.wavelets.ricker(freq, nt, dt, 1/freq)
                          .reshape(-1, 1, 1)
                          .repeat(1, num_shots, num_sources_per_shot))

#______________________________________________________________________________
# Propagator call and shot extraction
prop = deepwave.scalar.Propagator({'vp': model_true.to(device)}, dx)
receiver_amplitudes_true = prop(source_amplitudes_true.to(device),
                                x_s.to(device),
                                x_v.to(device), dt).cpu()

#______________________________________________________________________________
# Take all the shots, convert to 3D numpy array, 
# and resample with antialias in the time direction
allshotsresamp=scipy.signal.decimate(receiver_amplitudes_true[:,:].cpu().numpy(), 
                                 time_decim, n=None, ftype='iir', axis=0, zero_phase=True)
vmin, vmax = np.percentile(allshotsresamp[:,0], [1.5,98.5])
plt.imsave('vsp2.png',allshotsresamp[:,0],cmap=plt.cm.seismic, vmin=-vmax, vmax=vmax)
#np.savez(shotsout,allshotsresamp)# save numpy array to disk
#______________________________________________________________________________
# Export the shots to SEGY
#wrap.Numpy2Segy("FDM_",allshotsresamp, 1000*dt*time_decim)