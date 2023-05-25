import h5py
import argparse
import numpy as np
import metzger2017 as m17
from scipy.interpolate import interp1d
from scipy.integrate import fixed_quad

parser = argparse.ArgumentParser()
parser.add_argument('beta', help='velocity distribution power-law parameter')
args = parser.parse_args()

ms = np.logspace(-3, -1, 11)
vs = np.linspace(0.05, 0.30, 6)
ks = np.linspace(0, 30, 7)
ks[0] += 1 # change smallest opacity from 0 to 1
beta = int(args.beta)
k_cut = False
r = 3.086e18 # parsec to cm
r *= 10 # 10 pc for absolute magnitude

outdir = 'M17_no_engine_full_opacity'
hdf5_data_path = outdir + '/beta' + str(beta)
hdf5_data_path += '/hdf5_data'

counter = 0
total = len(ms)*len(vs)*len(ks)

# make the hdf5 files to store LCs/spectra
# see https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
# in each hdf5 file, two datasets: "param" and "lightcurves" or "spectra"
# this avoids any data loading assumptions wrt parameter organization
# append each new simulation to hdf5 file, will help with i/o tremendously once library size reaches O(100s)
# consider OO approach... is it necessary?

for m in ms:
    for v in vs:
        for k in ks:
            counter += 1
            if counter % 10 == 0: print(counter, '/', total)
            param = np.c_[m, v, k]
            tdays, Ltot, flux, T, R = m17.calc_lc(m, v, beta, k, kappa_cut=k_cut)
            # cut the last time as it results in nans
            tdays = tdays[1:-1]
            flux = flux[:, 1:-1]*1e-8 # scale output to per Angstrom, shape = [n_wavs x n_times] = [1024, 72]
            flux = flux.T    # swap shape to [n_times x n_wavs] = [72, 1024]
            wavelengths = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # cm to microns
            flux = np.insert(flux, 0, wavelengths, axis=0) # append wavelengths to be stored in spectra
            np.savetxt('{0}/beta{1}/spectra/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), flux, header='Wavelengths (microns) \t Flux (erg / s / cm^2 / Angstrom) in shape [[wavs + n_times] x n_wavs] = [73 x 1024]', fmt="%g")
            try:
                params = np.concatenate((params, param), axis=0)
                spec = np.concatenate((spec, flux[None, :]), axis=0)
            except NameError:
                spec = flux[None, :]
                params = param


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
            np.savetxt('{0}/beta{1}/lc_lums/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), np.c_[tdays, lc_bol_lum], header='Time (days) \t L_bol (erg / s)', fmt="%.3f \t %.6e")
            try:
                lums = np.concatenate((lums, lc_bol_lum[None, :]), axis=0)
            except NameError:
                lums = lc_bol_lum[None, :]
            

            # convert Lbol to mags
            # L = flux*4*pi*r**2 -> flux = L/(4*pi*r**2) -> log10(flux) = log10(L/4*pi*r**2) = log10(L) - log10(4*pi*r**2)
            lc_bol_mag = np.log10(lc_bol_lum) - np.log10(4*np.pi*r**2)
            lc_bol_mag = -48.6 - 2.5*lc_bol_mag
            np.savetxt('{0}/beta{1}/lc_mags/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *param[0]), np.c_[tdays, lc_bol_mag], header='Time (days) \t m_{AB, bol}', fmt="%.3f \t %.3f")
            try:
                mags = np.concatenate((mags, lc_bol_mag[None, :]), axis=0)
            except NameError:
                mags = lc_bol_mag[None, :]
            

# save spectra to hdf5
print('writing spectra to hdf5...')
h5f = h5py.File(hdf5_data_path+'/spectra.h5', 'a')
header = h5f.create_dataset('header', (1))
header.attrs['units_wavelength'] = 'microns'
header.attrs['units_spectral_energy_density'] = 'erg / s / cm^2 / Angstrom'
header.attrs['spectra_array_structure'] = '[N, 74, 1024]: N = number of simulations, 74 = [0: wavelengths, 1-73: observation times], 1024 = spectral energy density in relevant wavelength bin'
header.attrs['observation_times'] = np.array2string(np.logspace(np.log10(0.125), np.log10(128), 81)[:73], precision=4, max_line_width=50)
header.attrs['units_observation_times'] = 'days'
h5f.create_dataset('params', data=params, compression="gzip", chunks=True, maxshape=(None, 3)) 
h5f.create_dataset('spectra', data=spec, compression="gzip", chunks=True, maxshape=(None, 74, 1024))
h5f.close()

# save luminosity LCs to hdf5
print('writing lc_lums to hdf5...')
h5f = h5py.File(hdf5_data_path+'/lc_lums.h5', 'a')
header = h5f.create_dataset('header', (1))
header.attrs['units_lc_lums'] = 'erg / s'
header.attrs['lc_lums_array_structure'] = '[N, 73]: N = number of simulations, 73 = observation times'
header.attrs['observation_times'] = np.array2string(np.logspace(np.log10(0.125), np.log10(128), 81)[:73], precision=4, max_line_width=50)
header.attrs['units_observation_times'] = 'days'
header.attrs['bolometric'] = True
h5f.create_dataset('params', data=params, compression="gzip", chunks=True, maxshape=(None, 3)) 
h5f.create_dataset('lc_lums', data=lums, compression="gzip", chunks=True, maxshape=(None, 73))
h5f.close()

# save AB magnitude LCs to hdf5
print('writing lc_mags to hdf5...')
h5f = h5py.File(hdf5_data_path+'/lc_mags.h5', 'a')
header = h5f.create_dataset('header', (1))
header.attrs['units_lc_mags'] = 'absolute AB magnitude (AB magnitude at 10 pc)'
header.attrs['lc_mags_array_structure'] = '[N, 73]: N = number of simulations, 73 = observation times'
header.attrs['observation_times'] = np.array2string(np.logspace(np.log10(0.125), np.log10(128), 81)[:73], precision=4, max_line_width=50)
header.attrs['units_observation_times'] = 'days'
header.attrs['bolometric'] = True
h5f.create_dataset('params', data=params, compression="gzip", chunks=True, maxshape=(None, 3)) 
h5f.create_dataset('lc_mags', data=mags, compression="gzip", chunks=True, maxshape=(None, 73))
h5f.close()
