import h5py
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(210905)

data = h5py.File('M17_no_engine_full_opacity/beta1/hdf5_data/spectra.h5')
header = data['header']
for attr in header.attrs:
    print(attr, header.attrs[attr])
params = data['params']
spectra = data['spectra']

sim_idx = np.random.choice(range(params.shape[0]))
param = params[sim_idx]
spec = spectra[sim_idx]

t_idx = np.random.choice(range(spec.shape[0]-1))

plt.plot(spec[0, :], spec[t_idx, :], c='k')
plt.xscale('log')
plt.yscale('log')
plt.gca().set_ylim(bottom=1e-10)
plt.savefig('sanity_check_hdf5.pdf')
