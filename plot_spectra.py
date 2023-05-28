import numpy as np
import h5py
import matplotlib.pyplot as plt

path_to_hdf5 = 'M17_no_engine_full_opacity/beta3/hdf5_data'

h5_spec = h5py.File(path_to_hdf5+'/spectra.h5', 'r')
#lambdaobs = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # cm to microns

print([attr for attr in h5_spec['header'].attrs])
print(h5_spec['header'].attrs['observation_times'])
# size [74 x 1024]
spectra = h5_spec['spectra']

# wavelengths are first index of any simulation (in this case the 50th simulation)
wavelengths = spectra[49, 0, :]

spectra = spectra[:, -10, :]

r = (297e6/10) # 297 Mpc divided by 10 pc from absolute magnitude
print(r**2)
spectra /= r**2

for i in range(spectra.shape[0]):
    plt.plot(wavelengths, spectra[i], c='r', alpha=0.3)
plt.xscale('log')
plt.yscale('log')
plt.ylim([1e-24, 3e-21])
plt.xlim([2, 5])
plt.savefig('test_spectra.png') 

