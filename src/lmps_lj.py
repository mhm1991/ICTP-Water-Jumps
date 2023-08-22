#!/usr/bin/env python
# coding: utf-8

# In[154]:


input_filename = "NVT_water_T_300.xyz"
output_filename = "NVT_water_T_300_rev.xyz"

num_atoms = 3057
cos_line = 9
iter_=20000
try:
    sk = cos_line*(iter_+1) + num_atoms*iter_
    sk_t = 1*(iter_+1) + (num_atoms+7)*iter_
    data_now = np.loadtxt(input_filename,skiprows=sk, max_rows=num_atoms)
#%step_now = np.loadtxt(input_filename,skiprows=sk_t, max_rows=1)
    
    if iter_ == 20000:
        data_ = data_now
#         step_ = np.zeros(step_now.shape)
    else:
        data_ = np.concatenate((data_, data_now), axis =0)
#         step_ = np.concatenate(step_, step_now, axis =0)
    
    iter_ +=1
#     print(data_now.shape)
    print(iter_,data_now.shape)
except StopIteration:
    pass


# In[155]:


input_filename = "NVT_water_T_300.xyz"
output_filename = "NVT_water_T_300_rev.xyz"

num_atoms = 3057
cos_line = 9
iter_=20000
readlines(input_filename)
exit()
try:
    sk = cos_line*(iter_+1) + num_atoms*iter_
    sk_t = 1*(iter_+1) + (num_atoms+7)*iter_
    data_now = np.loadtxt(input_filename,skiprows=sk, max_rows=num_atoms)
#%step_now = np.loadtxt(input_filename,skiprows=sk_t, max_rows=1)
    
    if iter_ == 20000:
        data_ = data_now
#         step_ = np.zeros(step_now.shape)
    else:
        data_ = np.concatenate((data_, data_now), axis =0)
#         step_ = np.concatenate(step_, step_now, axis =0)
    
    iter_ +=1
#     print(data_now.shape)
    print(iter_,data_now.shape)
except StopIteration:
    pass


# In[ ]:


# Read the input .xyz file

# Extract time step lines and atom lines
time_step_lines = []
time_step = 0
atom_dict = dict()
atome_lines = []
capture_atoms = False
inside_ts = False
time_count = 0
num_atoms = 3057
sample_time_sacle = 100
outfile = open(output_filename, 'w')
n_a = 0
time_step_str = ""

with open(input_filename, 'r') as infile:

    for line in infile:
        if line.startswith("ITEM: TIMESTEP"):
            n_a = 0
            outfile.write("ITEM: TIMESTEP = \n")
            print("ITEM: TIMESTEP = ")
            for line in infile:
                #print(line)
                if line.startswith("ITEM: ATOMS"):
                    for line in infile:
                        print("&&"+line.strip()+"\n")
                        outfile.write(line.strip()+"\n")
                        if n_a > num_atoms:
                            break
                        n_a += 1
                if n_a > num_atoms:
                    break
            
            


# In[108]:


input_filename = "NVT_water_T_300.xyz"
output_filename = "NVT_water_T_300_rev.xyz"

# Read the input .xyz file

# Extract time step lines and atom lines
time_step_lines = []
time_step = 0
atom_dict = dict()
atome_lines = []
capture_atoms = False
inside_ts = False
time_count = 0
num_atoms = 3057
sample_time_sacle = 100
outfile = open(output_filename, 'w')
n_a = 0
time_step_str = ""

with open(input_filename, 'r') as infile:

    for line in infile:
        if line.startswith("ITEM: TIMESTEP"):
            n_a = 0
            for line in infile:
                if line.startswith("ITEM: ATOMS"):
                    for line in infile:
                        outfile.write(line.strip()+"\n")
                        if n_a > num_atoms:
                            break
                        n_a += 1
            
            #time_step_lines.append("ITEM TIMESTEP = " + str(time_step) +'\n')
            #time_step_str = "ITEM TIMESTEP = " + str(time_step) +'\n'
            #time_step_linese_step_num.append(lines[time_count+1].strip())
            capture_atoms = False
            outfile.write("ITEM: TIMESTEP = "+str(time_step)+"\n")
            time_step += sample_time_sacle


        elif line.startswith("ITEM: ATOMS"):
            capture_atoms = True
            n_a = 0

            for line in infile:
                #atom_lines.append(line.strip())
                
            #for line in atom_lines:
                outfile.write(line.strip()+"\n")
                if n_a > num_atoms:
                    break
                n_a += 1

        #elif capture_atoms:
        #    atom_lines.append(line.strip())

        time_count += 1
        #print(time_step_linese_step_num[:-1])
        #if len(time_step_linese_step_num[:-1])>0:
        #    print(time_step_linese_step_num[:-1][0])
    
        # Write the time step lines and atom lines to the output file
        #for line in time_step_lines:
        
        
        
        time_step_lines.clear()
        time_step_linese_step_num.clear()
        atom_lines.clear()
    
    
outfile.close()
# Write the time step lines and atom lines to the output file
#with open(output_filename, 'w') as outfile:
#    for line in time_step_lines:
#        outfile.write(line + '\n')
#    for line in time_step_linese_step_num:
#        outfile.write(line + '\n')

#    for line in atom_lines:
#        outfile.write(line + '\n')

print("Output file with time steps and atom configurations has been created.")


# In[100]:


output_file_path = "NVT_water_T_300_rev.xyz"
trajectories = {}
current_timestep = None
atom_data = []
numer_of_selected_atoms = 3057
total_time_steps_limit = 200000
offset_time_stps = 0
lines = []
count = 0
fs = 1 
n_a = 0

with open(output_file_path, 'r') as file:
    count = 0
    for line in file:
        if "ITEM: TIMESTEP" in line and count <= total_time_steps_limit - sample_time_sacle:
            lines.append(line + str(count))
            n_a = 0
            for line in file:
                #atom_lines.append(line.strip())
                
            #for line in atom_lines:
                #print(line)
                lines.append(line)
                if n_a > num_atoms:
                    break
                n_a += 1
            count += sample_time_sacle
            lines.append(line)
        elif count > total_time_steps_limit - sample_time_sacle:
            break
        #elif count <= total_time_steps_limit - sample_time_sacle:
            
        #    if "ITEM: TIMESTEP" in line:
                #lines.append(line)
        #        count += sample_time_sacle
            #if count % (fs*sample_time_sacle) == 5:        
        #    lines.append(line)


            #if "ITEM: TIMESTEP" in line:
            #    count += sample_time_sacle
            #lines.append(line)

        #    break
        #elif count < total_time_steps_limit - sample_time_sacle:
        #    lines.append(line)
        #    count += sample_time_sacle

print(lines[:100])

for i, line in enumerate(lines[:]):
    print(i , line)
    if line.startswith("ITEM: TIMESTEP"):
        #current_timestep = int(lines[i + 1].strip())
        if current_timestep is not None and current_timestep % (fs*sample_time_sacle) == 0:
            trajectories[current_timestep] = atom_data
            atom_data = []
        print(current_timestep)
        #if current_timestep > total_time_steps_limit + offset_time_stps:
        #    break
    
        num_atoms = numer_of_selected_atoms

        for j in range(num_atoms):
            #print(lines[i+j+1])
            #atom_id, atom_type, x, y, z = lines[i+j+1].split()
            #print(atom_id,atom_type,x,y,z)
            atom_data.append((atom_id, atom_type, x, y, z))
        #i += num_atoms

if current_timestep is not None:
    trajectories[current_timestep] = atom_data

# Print atom positions for each timestep
#time_step_vec = []
#for timestep, atom_data in trajectories.items():
#    print(f"TIMESTEP {timestep}")
    #for atom_id, atom_type, x, y, z in atom_data:
        #print(f"{atom_id} {atom_type} {x} {y} {z}")
#    time_step_vec.append(timestep)
        
atom_data_dict = trajectories
file.close()


# In[75]:


atom_data_dict


# In[51]:


output_file_path = "NVT_water_T_300_rev.xyz"
trajectories = {}
current_timestep = None
atom_data = []
numer_of_selected_atoms = 3057
total_time_steps_limit = 200000
offset_time_stps = 0
lines = []
count = 0
fs = 1 
n_a = 0

with open(output_file_path, 'r') as file:
    count = 0
    for line in file:
        #print(line)
        if "ITEM: TIMESTEP" in line and count <= total_time_steps_limit - sample_time_sacle:
            lines.append(line)
            n_a = 0
            count += sample_time_sacle

            for line in file:
                #print(line)
                #atom_lines.append(line.strip())
                
            #for line in atom_lines:
                lines.append(line)
                #print(line)
                if n_a >= num_atoms:
                    break
                n_a += 1
           
        #elif "ITEM: TIMESTEP" in line and count > total_time_steps_limit - sample_time_sacle:
        #    break
        

        #elif count <= total_time_steps_limit - sample_time_sacle:
            
        #    if "ITEM: TIMESTEP" in line:
                #lines.append(line)
        #        count += sample_time_sacle
            #if count % (fs*sample_time_sacle) == 5:        
        #    lines.append(line)


            #if "ITEM: TIMESTEP" in line:
            #    count += sample_time_sacle
            #lines.append(line)

        #    break
        #elif count < total_time_steps_limit - sample_time_sacle:
        #    lines.append(line)
        #    count += sample_time_sacle

#print(lines[:100])
for i, line in enumerate(lines):
    
    if line.startswith("ITEM: TIMESTEP"):
        current_timestep = int(lines[i + 1].strip())
        print(current_timestep)
        if current_timestep is not None and current_timestep % (fs*sample_time_sacle) == 0:
            trajectories[current_timestep] = atom_data
            atom_data = []
        print(current_timestep)
        #if current_timestep > total_time_steps_limit + offset_time_stps:
        #    break
    
        num_atoms = numer_of_selected_atoms

        for j in range(num_atoms):
            #print(lines[i+j+1])
            atom_id, atom_type, x, y, z = lines[i+j+1].split()
            #print(atom_id,atom_type,x,y,z)
            atom_data.append((atom_id, atom_type, x, y, z))
        i += 1

if current_timestep is not None:
    trajectories[current_timestep] = atom_data

# Print atom positions for each timestep
#time_step_vec = []
#for timestep, atom_data in trajectories.items():
#    print(f"TIMESTEP {timestep}")
    #for atom_id, atom_type, x, y, z in atom_data:
        #print(f"{atom_id} {atom_type} {x} {y} {z}")
#    time_step_vec.append(timestep)
        
atom_data_dict = trajectories
file.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[144]:


import numpy as np


# Convert the atom data dictionary to a NumPy array for faster computation
atom_data_array = {}
box_size = 31
Equi_time_steps = offset_time_stps
Prod_time_steps = total_time_steps_limit
time_step_max = Equi_time_steps + Prod_time_steps 
#Correct coordinates
def pbc(x,box_size):
    x = x - box_size*np.round(x/box_size)
    return x





for timestep, atom_list in atom_data_dict.items():
    #print(timestep,atom_list[1])
    atom_data_array[timestep] = np.array([[item for item in atom] for atom in atom_list])

# Dictionary to store dipole vectors for each time step
dipole_vectors = {}

whole_timesteps = []

# Loop through each time step and calculate dipole vectors using NumPy
for timestep, atom_array in atom_data_array.items():
    
    print(timestep)
    if len(atom_array) == 0:
        continue
    #print(atom_array)
    if timestep % 100 == 0:
        print(timestep)
    atom_array = np.array(atom_array)
    #print(atom_array)
    o_indices = np.where((atom_array[:, 1] == '1'))[0]# & (np.float(atom_array[:, 2]) < 10) & (np.float(atom_array[:, 2]) > -10) & (np.float(atom_array[:, 3]) < 10) & (np.float(atom_array[:, 3]) > -10) & (np.float(atom_array[:, 4]) < 10) & (np.float(atom_array[:, 4]) > -10))[0] 
    o_atoms = atom_array[o_indices, 2:].astype(float)
    h1_atoms = atom_array[o_indices + 1, 2:].astype(float)
    h2_atoms = atom_array[o_indices + 2, 2:].astype(float)
    
    
    oh1 = o_atoms - h1_atoms
    oh2 = o_atoms - h2_atoms
    oh1 = pbc(oh1, box_size)
    oh2 = pbc(oh2, box_size)
    
    # Calculate dipole vectors with the adjusted charge for oxygen
    #dipole_vector = oh1 + oh2#h1_atoms + h2_atoms - 2 * o_atoms
    #print(dipole_vector)

    dipole_vectors[timestep] = oh1 + oh2#dipole_vector/2
    #print(len(dipole_vectors[timestep]))
    #print(len(oh1 + oh2))
# Print dipole vectors for each time step
#for timestep, vectors in dipole_vectors.items():
#    print(f"Time step {timestep}:")
#    for vector in vectors:
#       print(f"Dipole vector: {vector}")
        
        


# In[ ]:





# In[169]:


ts = dipole_vectors.keys()
dp = []
dpx = []
dpy = []
dpz = []
for item in ts:
    dp.append(dipole_vectors[item][0])
    
for item in dp:
    dpx.append(item[0])
for item in dp:
    dpy.append(item[1])
#for item in dp:
#    dpz.append(item[2])
    
plt.plot(np.multiply(np.sign(dpx),np.exp(-np.abs(dpx))))
#plt.plot(dpx)
#plt.hist(np.sqrt(dpx**2+dpy**2+dpz**2))


# In[158]:


plt.hist(dpx)


# In[149]:


dipole_vectors.keys()


# In[134]:


dipole_matrix = np.array([dipole_vectors[ts] for ts in dipole_vectors.keys()])
dipole_concat = np.concatenate(dipole_matrix)
dipole_concat


# In[10]:


dipole_matrix


# In[135]:


dipole_matrix = np.reshape(dipole_matrix,(len(dipole_vectors.keys()),-1,3))

N_step = np.shape(dipole_matrix)[0]
N_atom = np.shape(dipole_matrix)[1]

print(N_step)


# # Case1:  correlation terms, cross correlations are excluded.
# 
# ## $C(\tau) = \sum_{i,j}\langle\vec{dp}_i(t_0)\cdot \vec{dp}_j(t_0+\tau) \rangle \delta_{ij}$

# In[182]:


from scipy.signal import correlate
import matplotlib.pyplot as plt

corr_atom = np.zeros((N_atom,N_step))
t_zero = np.zeros((N_atom,))

for i in range(N_atom):
    dipole_i_x = dipole_matrix[:,i,0]
    dipole_i_y = dipole_matrix[:,i,1]
    dipole_i_z = dipole_matrix[:,i,2]
    
    dipol_product_sqr = np.dot(dipole_i_x,dipole_i_x) + np.dot(dipole_i_y,dipole_i_y) + np.dot(dipole_i_z,dipole_i_z) 
    #dipole_abs = array_i[:,0] #np.sqrt(array_i[:,0]**2 + array_i[:,1]**2 + array_i[:,2]**2)#
    #print(dipole_abs.shape)
    #dipole_i = dipole_i_x + dipole_i_y + dipole_i_z
    correlation_x = correlate(dipole_i_x, dipole_i_x, mode='full')
    correlation_y = correlate(dipole_i_y, dipole_i_y, mode='full')
    correlation_z = correlate(dipole_i_z, dipole_i_z, mode='full')
    correlation = (correlation_x + correlation_y + correlation_z)/dipol_product_sqr
    corr_atom[i,:] = correlation[N_step - 1:2*N_step + 1]
    plt.plot(corr_atom[i,:])
    #print(correlation.shape)
    #corr_atom = np.array(corr_atom)
    corr = np.mean(corr_atom, axis = 0)
    indices = np.argwhere(corr_atom[i,:] < 0)
    try: 
        t_zero[i] = np.min(indices)
    except:
        continue
    
np.savetxt('corr300.txt',corr)  
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['text.usetex'] = True
plt.xlabel('Lag Time ($\\tau$) [100fs]')
plt.ylabel('Correlation Function')
plt.title('Normalize DP-DP Correlation Function in T = 300K and N = '+str(int(numer_of_selected_atoms/3))+' Molecules')
plt.grid(True)
plt.show()
plt.plot(corr)


    #corr_atom.append(correlation)
#plt.plot(np.mean(corr_atom))


# In[187]:


t_zero = np.nonzero(t_zero)
plt.hist(t_zero, bins=20)


# # Case 2: Cross-correlation terms included 
# ## $C(\tau) = \sum_{i,j}\langle\vec{dp}_i(t_0)\cdot \vec{dp}_j(t_0+\tau) \rangle$
# 

# In[ ]:





# In[142]:


from scipy.signal import correlate
import matplotlib.pyplot as plt

corr_atom = np.zeros(N_step)

dipole_x = np.sum(dipole_matrix[:,:,0],axis = 1)
dipole_y = np.sum(dipole_matrix[:,:,1],axis = 1)
dipole_z = np.sum(dipole_matrix[:,:,2],axis = 1)

#dipole_abs = array_i[:,0] #np.sqrt(array_i[:,0]**2 + array_i[:,1]**2 + array_i[:,2]**2)#
#print(dipole_abs.shape)
#dipole_i = dipole_i_x + dipole_i_y + dipole_i_z
correlation_x = correlate(dipole_x, dipole_x, mode='full')
correlation_y = correlate(dipole_y, dipole_y, mode='full')
correlation_z = correlate(dipole_z, dipole_z, mode='full')
dipol_product_sqr = np.dot(dipole_x,dipole_x) + np.dot(dipole_y,dipole_y) + np.dot(dipole_z,dipole_z) 

correlation = (correlation_x + correlation_y + correlation_z)/dipol_product_sqr
corr_atom[:] = correlation[N_step - 1:2*N_step + 1]
corr_range = range(len(np.log((corr_atom[:]))))
#plt.plot(range(len(corr_atom[:])),np.log(corr_atom))
#plt.plot(corr_range,np.log(corr_atom))

plt.plot(corr_atom)
#plt.plot(corr_atom[:],scalex= loglog, scaley=loglog)
#print(correlation.shape)
#corr_atom = np.array(corr_atom)
#corr = np.mean(corr_atom, axis = 0)

    

np.savetxt('corr_croos_300.txt',corr_atom)  
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
plt.rcParams['text.usetex'] = True
plt.xlabel('Lag Time ($\\tau$) [100fs]')
plt.ylabel('Correlation Function')
plt.title('Normalize DP-DP Correlation Function in T = 300K and N = '+str(int(numer_of_selected_atoms/3))+' Molecules')
plt.grid(True)
plt.show()

    #corr_atom.append(correlation)
#plt.plot(np.mean(corr_atom))


# In[ ]:





# In[ ]:





# In[137]:


import numpy as np
import matplotlib.pyplot as plt

# Replace this with your actual time series of correlation functions
# For demonstration purposes, let's generate a random time series
correlation_time_series = corr_atom#correlation[3800:0:-1]
total_lags = len(corr)#correlation[3800:0:-1])

# Apply Fast Fourier Transform (FFT)
correlation_spectrum = np.fft.fft(correlation_time_series)

# Calculate the corresponding frequency values
sampling_rate = 1.0  # Adjust this if your time series has a different sampling rate
frequencies = np.fft.fftfreq(total_lags, d=1/sampling_rate)

size_freq = len(frequencies)//2

# Plot the spectrum
plt.plot(frequencies[:size_freq], np.abs(correlation_spectrum)[:size_freq])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Correlation Spectrum")
plt.grid(True)
plt.xlim([0,0.05])
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

#numer_of_selected_atoms = 10
# Convert the dipole vectors dictionary to a NumPy array
dipole_vectors_array = {}
for timestep, vectors in dipole_vectors.items():
    #if timestep % 50 == 0:

    dipole_vectors_array[timestep] = np.array(vectors)


# Calculate the inner products of dipole vectors for each time step
dot_products = {}
for timestep, vectors in dipole_vectors_array.items():
    #if timestep % 50 == 0:

    dot_products[timestep] = np.dot(vectors, vectors.T)

# Calculate the correlation function for different lag times (\tau)
def calculate_correlation(tau):
    correlation = []
    print(tau)
    for timestep, dot_product in dot_products.items():
        #if timestep % 10 == 0:
        timestep_tau = timestep + tau
        #print(timestep_tau)
        #if timestep_tau in dot_products:
        #if timestep_tau < len(dipole_vectors_array):
        #if dipole_vectors_array[timestep_tau] != None:
        if timestep_tau < total_time_steps_limit - 10:
            corr = np.mean(dipole_vectors_array[timestep] * dipole_vectors_array[timestep_tau]) / dot_product
            correlation.append(corr)
    return np.mean(correlation)

# Define the range of lag times (\tau) you're interested in
tau_values = range(0, 000,50)  # Adjust the range as needed

# Calculate the correlation function for each lag time
correlation_function = [calculate_correlation(tau) for tau in tau_values]

# Plot the correlation function
plt.plot(tau_values, correlation_function, marker='o')
plt.xlabel('Lag Time (tau) [5fs]')
plt.ylabel('Normalized Correlation')
plt.title('Normalized DP-DP Correlation Function in T = 300K and N = '+str(int(numer_of_selected_atoms/3))+' Molecules')
plt.grid(True)
plt.show()


# In[17]:


import numpy as np
import matplotlib.pyplot as plt

# Replace this with your actual time series of correlation functions
# For demonstration purposes, let's generate a random time series
correlation_time_series = correlation_function
total_lags = len(correlation_function)

# Apply Fast Fourier Transform (FFT)
correlation_spectrum = np.fft.fft(correlation_time_series)

# Calculate the corresponding frequency values
sampling_rate = 1.0  # Adjust this if your time series has a different sampling rate
frequencies = np.fft.fftfreq(total_lags, d=1/sampling_rate)

# Plot the spectrum
plt.plot(frequencies, np.abs(correlation_spectrum))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Correlation Spectrum")
plt.grid(True)
plt.show()


# In[ ]:


for j in range(0, num_mol):
  x = A[3*j+0, t_start : ]
  y = A[3*j+1, t_start : ]
  z = A[3*j+2, t_start : ]
  xyz2 = np.array([x[:],y[:],z[:]])
  xyz = np.transpose(xyz2)

  fs = 1/t_step_for_filt
  fc = 1/100
  # Apply butterworth filtering here to obtain the filtered vector (fx, fy, fz)
  [b,a] = signal.butter(2, fc/(fs/2))

  fxyz = np.zeros([len(x), 3])
  fxyz[:,0] = signal.filtfilt(b, a, x)
  fxyz[:,1] = signal.filtfilt(b, a, y)
  fxyz[:,2] = signal.filtfilt(b, a, z)

  sxyz = np.zeros([len(fxyz[:,0]), 3])
  sxyz[:,0] = smooth(fxyz[:,0], int(100./t_step))
  sxyz[:,1] = smooth(fxyz[:,1], int(100./t_step))
  sxyz[:,2] = smooth(fxyz[:,2], int(100./t_step))

  for ii in range (0, tim):
    sxyz[ii,:]=sxyz[ii,:]/np.sqrt(np.dot(sxyz[ii,:],sxyz[ii,:]))

  # Compute the velocity and dot product
  vxyz = np.zeros([len(fxyz[:,0])-1 , 3])
  vxyz[:,0] = np.diff(sxyz[:,0])
  vxyz[:,1] = np.diff(sxyz[:,1])
  vxyz[:,2] = np.diff(sxyz[:,2])

  sxyz_1 = np.zeros([len(fxyz[:,0])-1, 3])
  sxyz_1[:,0] = sxyz[:-1,0]
  sxyz_1[:,1] = sxyz[:-1,1]
  sxyz_1[:,2] = sxyz[:-1,2]

  qxyz = np.cross(sxyz_1, vxyz)

  for ii in range (0, len(qxyz)):
    qxyz[ii,:]=qxyz[ii,:]/np.sqrt(np.dot(qxyz[ii,:],qxyz[ii,:]))

	# Compute the dot product between consecutive frames as indication of change of geodesic.
  ah = np.zeros([len(qxyz)-1])

  for i in range (0, len(qxyz)-1):
    factor = np.dot(qxyz[i,:],qxyz[i,:])*np.dot(qxyz[i+1,:],qxyz[i+1,:])
    ah[i] = 1.0 - np.dot(qxyz[i,:],qxyz[i+1,:])/np.sqrt(factor)

    if(1e3*ah[i]>0.1):
      counts[j]+=1

    if(ah[i] <= threshold):
      ah[i] = 0.0

  l = argrelextrema(ah, np.greater)
  t = np.array(l)

  # change in jump of angle , change in the angle
  ang = np.zeros(int(len(l[0])))
  dt = np.zeros(int(len(l[0])))
  mol = np.zeros(int(len(l[0])))

  ang[0] = np.dot(xyz[t[0,0],:], xyz[0,:])/np.sqrt(np.dot(xyz[t[0,0],:],xyz[t[0,0],:])*np.dot(xyz[0,:],xyz[0,:]))
  ang[0] = np.arccos(ang[0])
  dt[0] = t[0,0]
  mol =  np.full(int(len(l[0])), j, dtype=int)

  for i in range(1, len(l[0])):
    ang[i] = np.dot(xyz[t[0,i],:],xyz[t[0,i-1],:])/np.sqrt( np.dot( xyz[t[0,i],:],xyz[t[0,i],:] ) * np.dot(xyz[t[0,i-1],:],xyz[t[0,i-1],:]) )
    ang[i] = np.arccos(ang[i])
    dt[i] = t[0,i]-t[0,i-1]

  ang = ang*(180/np.pi)

  b_ang.append(ang[:])
  b_dt.append(dt[:])
  b_time.append(t[0,:])
  b_mol.append(mol[:])

b_ang_array = []
b_dt_array = []
b_time_array = []
b_mol_array = []

for item in b_ang:
  for i in item:
    b_ang_array.append(i)

for item in b_dt:
  for i in item:
    b_dt_array.append(i * t_step)

for item in b_time:
  for i in item:
    b_time_array.append(i * t_step)

for item in b_mol:
  for i in item:
    b_mol_array.append(i)

tbs = np.array([b_ang_array[:], b_dt_array[:], b_time_array[:],b_mol_array[:]])
tbs_transpose = np.transpose(tbs)
tbs_sort = tbs_transpose[tbs_transpose[:,2].argsort()]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




