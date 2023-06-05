import sys
import h5py
import argparse
import interpolators
import numpy as np
import metzger2017 as m17
from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad

def run_m17(params, beta):
    k_cut=False
    r = 3.086e18 # parsec to cm
    r *= 10 # 10 pc for absolute magnitude

    m, v, k = params
    tdays, Ltot, flux, T, R = m17.calc_lc(m, v, beta, k, kappa_cut=k_cut)
    # cut the last time as it results in nans
    tdays = tdays[1:-1]
    flux = flux[:, 1:-1]*1e-8 # scale output to per Angstrom, shape = [n_wavs x n_times] = [1024, 72]
    flux = flux.T    # swap shape to [n_times x n_wavs] = [72, 1024]
    wavelengths = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # cm to microns
    flux = np.insert(flux, 0, wavelengths, axis=0) # append wavelengths to be stored in spectra
    #np.savetxt('{0}/beta{1}/spectra/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), flux, header='Wavelengths (microns) \t Flux (erg / s / cm^2 / Angstrom) in shape [[wavs + n_times] x n_wavs] = [73 x 1024]', fmt="%g")
    spec = flux[None, :]

    # convert Flam to Lbol
    # first interpolate the spectrum across wavelengths
    lc_bol_lum = []
    # convert wavelengths to Angstroms to match F_lambda units of erg / s / cm^2 / Angstrom
    # this doesn't matter during plotting but DOES MATTER FOR INTEGRATION
    wavelengths *= 1e4 # microns to Angstroms
    for t in range(len(tdays)):
        # t + 1 to account for the fact that index 0 = wavelengths, not fluxes
        flux_smooth = interp1d(wavelengths, flux[t+1, :], kind='cubic')
        # integrate over wavelengths to get erg / s / cm^2
        F = fixed_quad(flux_smooth, wavelengths[0], wavelengths[-1])[0]
        # multiply by 4*pi*r^2 to get erg / s
        # r = 10 pc = 3.0857e19 cm
        Lbol = F * 4 * np.pi * r**2
        lc_bol_lum.append(Lbol)
    # convert to array for ease of operations in magnitude conversion
    lc_bol_lum = np.array(lc_bol_lum)
    #np.savetxt('{0}/beta{1}/lc_lums/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), np.c_[tdays, lc_bol_lum], header='Time (days) \t L_bol (erg / s)', fmt="%.3f \t %.6e")
    lums = lc_bol_lum[None, :]
    
    
    # convert Lbol to mags
    # L = flux*4*pi*r**2 -> flux = L/(4*pi*r**2) -> log10(flux) = log10(L/4*pi*r**2) = log10(L) - log10(4*pi*r**2)
    lc_bol_mag = np.log10(lc_bol_lum) - np.log10(4*np.pi*r**2)
    lc_bol_mag = -48.6 - 2.5*lc_bol_mag
    #np.savetxt('{0}/beta{1}/lc_mags/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), np.c_[tdays, lc_bol_mag], header='Time (days) \t m_{AB, bol}', fmt="%.3f \t %.3f")
    mags = lc_bol_mag[None, :]

    return spec, lums, mags

parser = argparse.ArgumentParser()
parser.add_argument('beta', type=int, help='velocity distribution power-law parameter')
args = parser.parse_args()

beta = int(args.beta)

# Initialize GP

GP = interpolators.GP()

path_to_hdf5 = 'm17_beta%d_kappabroad' % beta
#path_to_hdf5 = 'M17_no_engine_full_opacity/beta%d/hdf5_data' % beta
#path_to_hdf5 = 'test_h5' 

# Load hdf5 data for bolometric magnitude light curves

h5_mags = h5py.File(path_to_hdf5+'/lc_mags.h5', 'r')
params = h5_mags['params'][:]
lc_bol_mag = h5_mags['lc_mags'][:]
h5_mags.close()

print(params.shape, lc_bol_mag.shape)

# Train the GP

print('training GP...')

GP.train(params, lc_bol_mag, nro=0)

# Generate 1 million samples to evaluate for highest uncertainty

param_mins = [np.log10(0.01), 0.03, np.log10(0.05)]
param_maxs = [np.log10(0.1), 0.30, np.log10(100)]

test_params = np.random.uniform(low=param_mins, high=param_maxs, size=(int(1e5), 3))
test_params[:, 0] = 10**test_params[:, 0]
test_params[:, 2] = 10**test_params[:, 2]

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

# Run Metzger model

spec, lums, mags = run_m17(param_sim_to_place, beta)

if np.isnan(spec).any(): sys.exit()

# Save evolution of parameters and error as simulations are added

N_sim = params.shape[0]+1
max_sigma = test_std[idx_max_std]
f=open('error_evolution_beta%d.dat' % beta, 'a')
file_length = len(open('error_evolution_beta%d.dat' % beta, 'r').read().split('\n'))
if file_length == 1: f.writelines('N_sims \t m \t\t v \t\t kappa \t max_sigma\n')
f.writelines('{0:d} \t {1:.3f} \t {2:.3f} \t {3:.2f} \t {4:.3f}\n'.format(N_sim, *param_sim_to_place, max_sigma))
f.close()

# Update hdf5 file with new simulations
# Reopen in append mode, not read mode!

h5_mags = h5py.File(path_to_hdf5+'/lc_mags.h5', 'a')
h5_lums = h5py.File(path_to_hdf5+'/lc_lums.h5', 'a')
h5_spec = h5py.File(path_to_hdf5+'/spectra.h5', 'a')

print(param_sim_to_place.shape[0])

param_sim_to_place = param_sim_to_place.reshape(1, 3)

with h5py.File(path_to_hdf5+'/lc_mags.h5', 'a') as h5_mags:
    h5_mags['params'].resize((h5_mags['params'].shape[0] + param_sim_to_place.shape[0]), axis=0)
    h5_mags['params'][-param_sim_to_place.shape[0]:] = param_sim_to_place
    print(h5_mags['params'].shape)

    h5_mags['lc_mags'].resize((h5_mags['lc_mags'].shape[0] + mags.shape[0]), axis=0)
    h5_mags['lc_mags'][-mags.shape[0]:] = mags
    print(h5_mags['lc_mags'].shape)
    h5_mags.close()

with h5py.File(path_to_hdf5+'/lc_lums.h5', 'a') as h5_lums:
    h5_lums['params'].resize((h5_lums['params'].shape[0] + param_sim_to_place.shape[0]), axis=0)
    h5_lums['params'][-param_sim_to_place.shape[0]:] = param_sim_to_place
    print(h5_lums['params'].shape)

    h5_lums['lc_lums'].resize((h5_lums['lc_lums'].shape[0] + lums.shape[0]), axis=0)
    h5_lums['lc_lums'][-lums.shape[0]:] = lums
    print(h5_lums['lc_lums'].shape)
    h5_lums.close()

with h5py.File(path_to_hdf5+'/spectra.h5', 'a') as h5_spec:
    h5_spec['params'].resize((h5_spec['params'].shape[0] + param_sim_to_place.shape[0]), axis=0)
    h5_spec['params'][-param_sim_to_place.shape[0]:] = param_sim_to_place
    print(h5_spec['params'].shape)

    h5_spec['spectra'].resize((h5_spec['spectra'].shape[0] + spec.shape[0]), axis=0)
    h5_spec['spectra'][-spec.shape[0]:] = spec
    print(h5_spec['spectra'].shape)
    h5_spec.close()
