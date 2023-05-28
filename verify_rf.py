import numpy as np
import joblib
import matplotlib.pyplot as plt
import metzger2017 as m17

path_to_hdf5 = 'M17_no_engine_full_opacity/beta3/hdf5_data'
rf = joblib.load(path_to_hdf5+'/../emulator/metzger_rf.joblib')

wavelengths = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # cm to microns
mask = np.where((wavelengths>2) & (wavelengths<5))[0]
wavelengths = wavelengths[mask]
test_params = np.array([np.log10(0.06), 0.06, 12, np.log10(29.344)])
test_params = test_params[None, :]

test_spec = 10**rf.evaluate(test_params).flatten()

tdays, _, reference_spec, _, _ = m17.calc_lc(0.06, 0.06, 1, 12)

time_idx = np.argmin(np.abs(tdays-28.9))
print(tdays[time_idx])

reference_spec = reference_spec[mask, time_idx+1]*1e-8

r = (297e6/10) # 297 Mpc divided by 10 pc from absolute magnitude
test_spec /= r**2
reference_spec /= r**2

plt.plot(wavelengths, reference_spec, c='k', label='Metz')
plt.plot(wavelengths, test_spec, c='r', label='intp')
plt.legend()
plt.xlim([2, 5])
plt.ylim([1e-22, 1.6e-20])
plt.xscale('log')
plt.yscale('log')
plt.savefig('test_rf_spec.png')
