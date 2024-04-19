import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Directory of the catalog: where the catalog is located
cat_eazy = fits.open('gds-grizli-v6.1-fix.eazypy.zout.fits')
# get the catalog data
cat_eazy_data = cat_eazy[1].data

# Right ascension and declination
ra = cat_eazy_data['ra']
dec = cat_eazy_data['dec']

# redshifts: spectroscopic and photometric redshifts
z_spec = cat_eazy_data['z_spec']
z_phot = cat_eazy_data['z_phot']

# rest-frame magnitudes in U, V, and J bands
restU = cat_eazy_data['restU']
restV = cat_eazy_data['restV']
restJ = cat_eazy_data['restJ']

# stellar mass (SM) and star formation rate (SFR)
SM = cat_eazy_data['mass']
SFR = cat_eazy_data['sfr']

# number of objects (i.e., galaxies)
ngals = len(ra)

cat_eazy.close()

cat_photo = fits.open('gds-grizli-v6.1-fix_phot_apcorr.fits')
# get the catalog data
cat_photo_data = cat_photo[1].data
cat_photo.close()

# check the list of filters and their names from the header information printed above
filter_names = ['f090w', 'f105w', 'f110w', 'f115w', 'f125w', 'f140w', 'f150w', 'f160w', 'f182m', 'f200w', 'f210m', 'f277w', 'f335m', 'f356w', 'f410m', 'f430m', 'f435w', 'f444w', 'f460m', 'f475w', 'f480m', 'f606w', 'f606wu', 'f775w', 'f814w', 'f814wu', 'f850lp', 'f850lpu']

nfilters = len(filter_names)

# get central wavelength of the filters
filter_cwave = np.zeros(nfilters)
for bb in range(nfilters):
    filter_cwave[bb] = cat_photo[1].header[filter_names[bb]+'_PLAM']
    
# get the fluxes in units of micro Jansky (uJy)
# The measurements were done with circular aperture with 0.7 arcsecond diameter
# Now the fluxes have been corrected for the flux loss due to the small aperture 
flux = {}
flux_err = {}
for bb in range(nfilters):
    flux[filter_names[bb]] = np.zeros(ngals)
    flux_err[filter_names[bb]] = np.zeros(ngals)
    
for ii in range(ngals):
    for bb in range(nfilters):
        #flux[filter_names[bb]][ii] = cat_photo_data[filter_names[bb]+'_flux_aper_2'][ii]
        #flux_err[filter_names[bb]][ii] = cat_photo_data[filter_names[bb]+'_fluxerr_aper_2'][ii]
        flux[filter_names[bb]][ii] = cat_photo_data[filter_names[bb]+'_tot_2'][ii]
        flux_err[filter_names[bb]][ii] = cat_photo_data[filter_names[bb]+'_etot_2'][ii]
        
def H(z):
    H0 = 70
    omega0_m = 0.3
    omega0_de = 0.7
    a = 1 / (1+z)
    H = H0 * np.sqrt(omega0_m * a**(-3) + omega0_de) # in km/s/MPC
    H = H * 1000 / (3.086e22) # in s^-1
    H = H * 365.25 * 24 * 3600
    return H

not_quiescent = [98, 699, 1527, 1554, 3328, 6776, 6777, 6836, 7004, 7353, 11471, 14380, 15073, 15441, 15522, 16236, 16406, 17744, 17967, 18442, 18518, 19768, 20458, 20820, 22100, 23906, 24237, 25692, 27165, 30071, 32225, 32226, 33671, 33822, 33934, 35368, 36150, 37097, 37666, 38624, 42185, 43016, 43814, 45186, 45905, 47476, 48634, 49134, 49696, 50144]

idx_sel_final = np.where((z_phot > 2.0) & (z_phot < 7.0) & (-2.5*np.log10(restU / restV) > 1.1) & (-2.5*np.log10(restV / restJ) < 1.8) & (-2.5*np.log10(restU / restV) > 0.88 * -2.5*np.log10(restV / restJ) + 0.29))

ids_sample_galaxies = [galaxy_id for galaxy_id in idx_sel_final[0] if galaxy_id not in not_quiescent]

ngals_sample = len(ids_sample_galaxies)

# First, we need to sort the filters from shortest to longest wavelength
sort_idx = np.argsort(filter_cwave)

sorted_filter_names = [] 
sorted_filter_cwave = []
for bb in range(nfilters):
    sorted_filter_names.append(filter_names[sort_idx[bb]])
    sorted_filter_cwave.append(filter_cwave[sort_idx[bb]])
    
sorted_filter_cwave = np.asarray(sorted_filter_cwave)

# get the SEDs (which is fluxes across the filters/wavelength)
gal_SED_flux = np.zeros((ngals_sample,nfilters))
gal_SED_flux_err = np.zeros((ngals_sample,nfilters))
for ii in range(ngals_sample):
    idx_gal = ids_sample_galaxies[ii]
    for bb in range(nfilters):
        gal_SED_flux[ii][bb] = flux[sorted_filter_names[bb]][int(idx_gal)]
        gal_SED_flux_err[ii][bb] = flux_err[sorted_filter_names[bb]][int(idx_gal)]
        
## plot the SEDs of the galaxies
for ii in range(ngals_sample):
    
    idx_wave = np.where((gal_SED_flux[ii]>-1) & (np.isnan(gal_SED_flux[ii])==False))

# Set the list of filter names
filters = ['hst_acs_f435w', 'hst_acs_f475w', 'hst_acs_f606w', 'hst_acs_f606w', 'hst_acs_f775w', 'hst_acs_f814w', 'hst_acs_f814w', 'jwst_nircam_f090w', 'hst_acs_f850lp', 'hst_wfpc2_f850lp', 'hst_wfc3_ir_f105w', 'hst_wfc3_ir_f110w', 'jwst_nircam_f115w', 'hst_wfc3_ir_f125w', 'hst_wfc3_ir_f140w', 'jwst_nircam_f150w', 'hst_wfc3_ir_f160w', 'jwst_nircam_f182m', 'jwst_nircam_f200w', 'jwst_nircam_f210m', 'jwst_nircam_f277w', 'jwst_nircam_f335m', 'jwst_nircam_f356w', 'jwst_nircam_f410m', 'jwst_nircam_f430m', 'jwst_nircam_f444w', 'jwst_nircam_f460m', 'jwst_nircam_f480m']


from astropy.cosmology import FlatLambdaCDM
from piXedfit.piXedfit_model import save_models_rest_spec

imf_type = 1                    # Chabrier (2003)
sfh_form = 4                    # double power-law SFH model
dust_law = 1                    # Calzetti (2000) dust attenuation law
duste_switch = 0                # turn off dust emission
add_neb_emission = 1            # turn on nebular emission
add_agn = 0                     # turn off AGN dusty torus emission

nmodels = 50000                 # number of model spectra to be produced
nproc = 8                       # number of cores to be used in the calculation


for i in range(2, 7):
    min_z = i                       # minimum redshift which determines the maximum age of the models
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    age_univ = cosmo.age(min_z)
    max_log_age = np.log10(age_univ.value)

    # we fix the ionization parameter to log(U)=-2.0
    params_range = {'dust2':[0.0,4.0], 'log_age':[-2.0,max_log_age], 'log_tau':[-1.0,1.5], 
                    'gas_logu':[-4.0,-1.0], 'log_alpha':[-1.0,1.0], 'log_beta':[-1.0,1.0]}

    '''
    models_spec = 'pixedfit_model_specs_minz%.2f.hdf5' % min_z
    save_models_rest_spec(imf_type=imf_type, sfh_form=sfh_form, dust_law=dust_law, params_range=params_range,
                            duste_switch=duste_switch, add_neb_emission=add_neb_emission, add_agn=add_agn,
                            nmodels=nmodels, nproc=nproc, name_out=models_spec)
    '''
    #ID of the first galaxy in the selected sample
    for ii in range(len(ids_sample_galaxies)):
        idx_gal_cat = ids_sample_galaxies[ii]
        
        if (z_phot[idx_gal_cat] >= i) and (z_phot[idx_gal_cat] <= i+1):

            # input SED: this example for first selected galaxy
            ## piXedfit only accept fluxes in unit of erg/s/cm^2/A, so we need to convert our fluxes from microjansky to this unit
            # first we need to get the central wavelength of the filters in Angstrom. We can use the following piXedfit's function
            from piXedfit.utils.filtering import cwave_filters
            photo_wave = cwave_filters(filters)

            from piXedfit.piXedfit_images import convert_flux_unit
            obs_flux = convert_flux_unit(photo_wave,gal_SED_flux[ii],init_unit='uJy',final_unit='erg/s/cm2/A')
            obs_flux_err = convert_flux_unit(photo_wave,gal_SED_flux_err[ii],init_unit='uJy',final_unit='erg/s/cm2/A')

            # the galaxy's redshift
            gal_z = z_phot[int(idx_gal_cat)]

            # Define priors
            from piXedfit.piXedfit_fitting import priors

            # Define the ranges of the parameters
            from astropy.cosmology import FlatLambdaCDM
            cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
            age_univ = cosmo.age(gal_z)
            max_log_age1 = np.log10(age_univ.value)

            ranges = {'dust2':[0.0,4.0], 'log_age':[-2.0,max_log_age1], 'log_tau':[-1.0,1.5], 
                      'gas_logu':[-4.0,-1.0], 'log_alpha':[-1.0,1.0], 'log_beta':[-1.0,1.0]}
            pr = priors(ranges)
            params_ranges = pr.params_ranges()

            # define the shape of the priors. Here we use uniform (i.e., flat) prior over the defined ranges.
            prior1 = pr.uniform('dust2')
            prior2 = pr.uniform('log_age')
            prior3 = pr.uniform('log_tau')
            prior4 = pr.uniform('gas_logu')
            prior5 = pr.uniform('log_alpha')
            prior6 = pr.uniform('log_beta')
            params_priors = [prior1, prior2, prior3, prior4, prior5, prior6]

            from piXedfit.piXedfit_fitting import singleSEDfit

            add_igm_absorption = 1
            igm_type = 1

            fit_method = 'mcmc'
            nwalkers = 100
            nsteps = 600

            nproc = 5                # number of cores to be used in the calculation

            name_out_fits = '%d_fresco_pixedfit_mcmc_%d.fits' % (i,idx_gal_cat+1)
            singleSEDfit(obs_flux=obs_flux, obs_flux_err=obs_flux_err, filters=filters, gal_z=gal_z, 
                         models_spec='pixedfit_model_specs_minz%.2f.hdf5' % min_z, params_ranges=params_ranges, params_priors=params_priors, 
                         fit_method=fit_method, add_igm_absorption=add_igm_absorption, igm_type=igm_type, 
                         nwalkers=nwalkers, nsteps=nsteps, nproc=nproc, store_full_samplers=1, name_out_fits=name_out_fits)