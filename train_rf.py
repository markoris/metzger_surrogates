import h5py
import joblib
import numpy as np
import interpolators
import data_format_1d as df1d
import matplotlib.pyplot as plt

rf = interpolators.RF()

# load hdf5 data
path_to_hdf5 = 'M17_no_engine_full_opacity/beta3/hdf5_data'
h5_spec = h5py.File(path_to_hdf5+'/spectra.h5', 'r')
params = h5_spec['params']
spectra = h5_spec['spectra']

# setting up times array
times = np.logspace(np.log10(0.125), np.log10(128), 81)[:74] # captures t = 64 days without excess that goes out to 128 days
times = times[:-10]
print(times[-1])

# get wavelengths
wavelengths = spectra[49, 0, :] # N_sims, wavs++N_times, N_lams
# go out to 29 days
spectra = spectra[:, 1:-9, :]

# append time as training parameter
params, spectra = df1d.format(params, spectra, times, 1)
print(params.shape, spectra.shape)

# trim wavelengths between 2 and 5 microns
mask = np.where((wavelengths > 2) & (wavelengths < 5))[0]
spectra = spectra[:, mask]
spectra = np.log10(spectra)

# take log mass and log time due to dynamic range of data
params[:, 0] = np.log10(params[:, 0])
params[:, 3] = np.log10(params[:, 3])

print(spectra.shape)

print('training rf... ETA ~4 mins')
rf.train(params, spectra)

joblib.dump(rf, path_to_hdf5+'/../emulator/metzger_rf.joblib')

