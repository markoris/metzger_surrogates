import argparse
import numpy as np
import metzger2017 as m17

parser = argparse.ArgumentParser()
parser.add_argument('beta', help='velocity distribution power-law parameter')
args = parser.parse_args()

ms = np.logspace(-3, -1, 11)
vs = np.linspace(0.05, 0.30, 6)
ks = np.linspace(0, 30, 7)
ks[0] += 1 # change smallest opacity from 0 to 1
beta = int(args.beta)
k_cut = False

outdir = 'M17_no_engine_full_opacity'

counter = 1
total = len(ms)*len(vs)*len(ks)

# make the hdf5 files to store LCs/spectra
# see https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
# in each hdf5 file, two datasets: "params" and "lightcurves" or "spectra"
# this avoids any data loading assumptions wrt parameter organization
# append each new simulation to hdf5 file, will help with i/o tremendously once library size reaches O(100s)
# consider OO approach... is it necessary?

for m in ms:
    for v in vs:
        for k in ks:
            counter += 1
            if counter % 10 == 0: print(counter, '/', total)
            params = np.c_[m, v, k]
            tdays, Ltot, flux, T, R = m17.calc_lc(m, v, beta, k, kappa_cut=k_cut)
            # cut the last time as it results in nans
            flux = flux[:, :-1]*1e-8 # scale output to per Angstrom, shape = [n_wavs x n_times] = [1024, 72]
            flux = flux.T    # swap shape to [n_times x n_wavs] = [72, 1024]
            wavelengths = np.logspace(np.log10(1e-5), np.log10(1.28e-3), 1024)*1e4 # cm to microns
            flux = np.insert(flux, 0, wavelengths, axis=0) # append wavelengths to be stored in spectra
            np.savetxt('{0}/beta{1}/spectra/m{2:.4f}_v{3:.2f}_kappa{4:g}.dat'.format(outdir, beta, *params[0]), flux, header='[[wavs + n_times] x n_wavs] = [73 x 1024]') 
            # convert Flam to Lbol
            
            # convert Lbol to mags
            # save Flam, mags
