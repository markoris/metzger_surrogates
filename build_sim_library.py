import h5py
import interpolators
import numpy as np

# Initialize GP

GP = interpolators.GP()

path_to_lc_bol_mag = 'M17_no_engine_full_opacity/beta1/hdf5_data/lc_mags.h5' 

# Load hdf5 data for bolometric magnitude light curves

h5_load = h5py.File(path_to_lc_bol_mag, 'r')
params = h5_load['params'][:]
lc_bol_mag = h5_load['lc_mags'][:]

# Train the GP

print('training GP...')

GP.train(params, lc_bol_mag, nro=0)

# Generate 1 million samples to evaluate for highest uncertainty

param_mins = [-3, 0.01, 1]
param_maxs = [-1, 0.30, 30]

test_params = np.random.uniform(low=param_mins, high=param_maxs, size=(int(1e5), 3))
test_params[:, 0] = 10**test_params[:, 0]

# Evaluate test samples

print('evaluating test samples...')

test_out, test_std = GP.evaluate(test_params)

# Average uncertainty across all times 
# to produce 1 million mean sigma values

test_std = np.mean(test_std, axis=1)

# Find highest-uncertainty candidate

idx_max_std = np.argmax(test_std)

print('highest uncertainty (in bol AB mag): ', test_std[idx_max_std])

param_sim_to_place = test_params[idx_max_std]

print('place simulation at these parameters: ', param_sim_to_place)

# Save evolution of parameters and error as simulations are added

N_sim = params.shape[0]+1
new_sim_params = param_sim_to_place
max_sigma = test_std[idx_max_std]
f=open('error_evolution.dat', 'a')
f.writelines('{0} \t {1} \t {2} \t {3} \t {4}'.format(N_sim, *new_sim_params, max_sigma))
f.close()
