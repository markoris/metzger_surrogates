def load_data():
	import numpy as np
	import matplotlib.pyplot as plt
	
	lums_data = np.loadtxt('/home/marko.ristic/lanl/knsc1_active_learning/Run_TP_dyn_all_lanth_wind2_all_md0.100000_vd0.052516_mw0.000457_vw0.294446_lums_2020-04-22.dat')
	mags_data = np.loadtxt('/home/marko.ristic/lanl/knsc1_active_learning/Run_TP_dyn_all_lanth_wind2_all_md0.100000_vd0.052516_mw0.000457_vw0.294446_mags_2020-04-22.dat')
	spec_data = np.loadtxt('/home/marko.ristic/lanl/knsc1_active_learning/Run_TP_dyn_all_lanth_wind2_all_md0.100000_vd0.052516_mw0.000457_vw0.294446_spec_2020-04-22.dat')

	lums_data = np.array(np.split(lums_data, 10)) # split across 10 wavelength bands: bolometric, g, r, i, z, y, J, H, K, S
	times_lums = lums_data[0, :, 1]
	lums = lums_data[:, :, 2:]

	mags_data = np.array(np.split(mags_data, 9)) # split across 9 wavelength bands: g, r, i, z, y, J, H, K, S. magnitudes do not include a bolometric component
	times_mags = mags_data[0, :, 1] # for any band, arbitrarily using the first one for convenience, take all the times at which magnitudes are reported
	mags = mags_data[:, :, 2:] # first index of 3rd axis are iterations, second are time points, the rest are data for 54 angular bins

	f = open('/home/marko.ristic/lanl/knsc1_active_learning/Run_TP_dyn_all_lanth_wind2_all_md0.100000_vd0.052516_mw0.000457_vw0.294446_spec_2020-04-22.dat', 'r')
	lines = f.readlines()
	for line in lines:
		if line[0] == '#': # time information is included in the header lines
			try:
				times_spec = np.append(times_spec, float(line.split()[-1]))
			except NameError:
				times_spec = float(line.split()[-1])
	f.close()

	spec_data = np.array(np.split(spec_data, spec_data.shape[0]/1024)) # each spectrum has 1024 wavelength bins, thus number of days = total shape / 1024 wav bins
	wav_lower = spec_data[0, :, 0] # lower limit of wavelength bins
	wav_upper = spec_data[0, :, 1] # upper limit of wavelength bins
	spec = spec_data[:, :, 2:]

	return times_lums, times_mags, times_spec, lums, mags, spec, wav_lower, wav_upper

def filter_parse(band):
	'''
	returns wavelengths and transmission efficiency for given filter
	'''
	import numpy as np

	filters = {
	'g': 'filters/g_LSST.dat',
	'r': 'filters/r_LSST.dat',
	'i': 'filters/i_LSST.dat', 
	'z': 'filters/z_LSST.dat',
	'y': 'filters/y_LSST.dat',
	'J': 'filters/J_2MASS.dat',
	'H': 'filters/H_2MASS.dat',
	'K': 'filters/K_2MASS.dat',
	'S': 'filters/Spitzer_IRAC2.dat'}

	return np.loadtxt(filters[band])

def lums_to_mags(lums):
	'''
	converts luminosities, given in ergs/s, to absolute magnitudes (calculated at 10 pc)
	'''
	import numpy as np

	lums = np.copy(lums)

	d = 3.086e18 	# parsec in cm
	d *= 10 	# distance of 10 pc for absolute magnitude
	flux = lums / (4 * np.pi * d**2)
	mags = np.where(flux > 0, -48.6 - 2.5*np.log10(flux), 0)
	return mags

def mags_to_lums(mags):
	'''
	converts magnitudes, given in absolute magnitudes, to luminosities
	'''
	import numpy as np
	
	mags = np.copy(mags)

	d = 3.086e18 	# parsec in cm
	d *= 10 	# distance of 10 pc

	mags += 48.6 
	mags /= -2.5
	flux = 10**(mags)
	lums = flux * (4 * np.pi * d**2)
	return lums

def spec_to_mags(wavs, spec, band):
	'''
	converts spectra to magnitudes, update this to be more informative later
	relies heavily on Eve Chase's code implementation, a big thank you to her for the insights and help provided
	'''
	import numpy as np
	from scipy.interpolate import interp1d
	from scipy.integrate import fixed_quad

	wavs, spec = np.copy(wavs), np.copy(spec)

	z = 0.0098 # distance to NGC 4993, host galaxy of AT2017gfo

	filter_data = filter_parse(band) # starts as transmission as a function of wavelength in ANGSTROMS
	filter_data[:, 0] *= 1e-8 # filter wavelengths from Angstrom to cm

	spec *= 1e8 # spectral flux density starts as erg/s/Angstrom/cm^2 -> 10^8 Anstroms per cm -> spectral flux density now in units of erg/s/cm/cm^2
		    # wavs still in units of cm
	
	wav_lower = filter_data[0, 0]
	wav_upper = filter_data[-1, 0]

	filt = interp1d(filter_data[:, 0], filter_data[:, 1], fill_value=1e-30) # wavelengths in cm
	
	spectrum = interp1d(wavs, spec, fill_value=1e-30) # wavelengths in cm, flux in erg/s/cm^2/cm

	numerator = lambda wav : wav/(1+z) * spectrum(wav/(1+z)) * filt(wav)
	denominator = lambda wav : filt(wav)/wav
	
	num = fixed_quad(numerator, wav_lower, wav_upper)[0]
	den = fixed_quad(denominator, wav_lower, wav_upper)[0]

	flux = num/den/3e10*54 # factor of 54 
	return np.array([-48.6 - 2.5*np.log10(flux), flux])

def spec_to_lums(wavs, spec, band):
	mags = spec_to_mags(wavs, spec, band)[0]
	return mags_to_lums(mags)
