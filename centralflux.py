#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import foale.mnsa
import fitsio
import time

# initialize data for all galaxies in survey
version = 'dr17-v0.1'
summary_file = os.path.join(os.getenv('MNSA_DATA'), version,'mn-{v}-summary.fits')
summary_file = summary_file.format(v=version)
f = fitsio.read(summary_file)
header = fitsio.read(summary_file, header = True)

# initialize Pipe3D data
pipe3D = fitsio.FITS('SDSS17Pipe3D_v3_1_1.fits')
pipe3D_galproperties = pipe3D[1]

# extract plateifu for each galaxy
MNSA_plateifu = f['plateifu']

# match the plateifus from Pipe3D with MNSA
pipe3D_plateifu = pipe3D_galproperties['plateifu'][:]
pipe3D_index = np.zeros(len(pipe3D_plateifu), dtype = int)

for i, p_ifu in enumerate(pipe3D_plateifu):
    pipe3D_index[i] = np.where(p_ifu == MNSA_plateifu)[0][0]

print(MNSA_plateifu[pipe3D_index] == pipe3D_plateifu)

def central_flux(p_IFU):
    """Function to check whether a galaxy has a DAP file associated with it and
       calculates central flux of each emission line of interest for each good galaxy"""
    # Read in the cube's FITS file
    m = foale.mnsa.MNSA(plateifu = p_IFU, dr17 = True)
    m.read_cube()

    try:
        m.read_maps()
        success = True
    except OSError:
        #print("Map reading failed! Skipping this case")
        success = False
        
    if(success is True):
        wave = m.cube['WAVE'].read()
        flux = m.cube['FLUX'].read()
        PSF = m.cube['RPSF'].read()        
        
        # Read in the emission line measurements (Gaussian fit)
        # This is a big 3D array, with one 2D image per line.
        gflux = m.maps['EMLINE_GFLUX'].read()

        # Pick out emission lines of interest
        halpha_indx = m.lineindx['Ha-6564']
        hbeta_indx = m.lineindx['Hb-4862']
        SII_6718_indx = m.lineindx['SII-6718']
        SII_6732_indx = m.lineindx['SII-6732']
        OI_indx = m.lineindx['OI-6302']
        OIII_indx = m.lineindx['OIII-5008']
        NII_indx = m.lineindx['NII-6585']

        # DAP Maps for Emission Lines of Interest
        halpha = gflux[halpha_indx, :, :]
        hbeta = gflux[hbeta_indx, :, :]
        SII_6718 = gflux[SII_6718_indx, :, :]
        SII_6732 = gflux[SII_6732_indx, :, :]
        OI = gflux[OI_indx, :, :]
        OIII = gflux[OIII_indx, :, :]
        NII = gflux[NII_indx, :, :]
        
        # Inverse variance maps for each line       
        gflux_ivar = m.maps['EMLINE_GFLUX_IVAR'].read()
        halpha_ivar = gflux_ivar[halpha_indx, :, :]
        hbeta_ivar = gflux_ivar[hbeta_indx, :, :]
        SII_6718_ivar = gflux_ivar[SII_6718_indx, :, :]
        SII_6732_ivar = gflux_ivar[SII_6732_indx,:,:]
        OI_ivar = gflux_ivar[OI_indx, :, :]
        OIII_ivar = gflux_ivar[OIII_indx,:,:]
        NII_ivar = gflux_ivar[NII_indx,:,:]
        
        # masked DAP maps
        gfluxmask = m.maps['EMLINE_GFLUX_MASK'].read()
        halpha_mask = np.ma.array(halpha, mask = gfluxmask[halpha_indx,:,:])
        hbeta_mask = np.ma.array(hbeta, mask = gfluxmask[hbeta_indx,:,:])
        SII_6718_mask = np.ma.array(SII_6718, mask = gfluxmask[SII_6718_indx,:,:])
        SII_6732_mask = np.ma.array(SII_6732, mask = gfluxmask[SII_6732_indx,:,:])
        OI_mask = np.ma.array(OI, mask = gfluxmask[OI_indx,:,:])
        OIII_mask = np.ma.array(OIII, mask = gfluxmask[OIII_indx,:,:])
        NII_mask = np.ma.array(NII, mask = gfluxmask[NII_indx,:,:])

        # don't want divide by zero errors when we calculate line ratios; make a mask that is comprised of good pixels AND exludes flux values less than or equal to zero: 
        # bit number = 0 --> no coverage in cube
        m = (halpha_mask == 0) | (hbeta_mask == 0 ) | (SII_6718_mask == 0) |(SII_6732_mask ==0) | (OI_mask == 0) |(OIII_mask == 0) | (NII_mask == 0) |(halpha <= 0) | (hbeta <= 0) |(SII_6718 <= 0 ) | (SII_6732 <= 0 ) |(OI <= 0) |(OIII <= 0) |(NII <= 0)

        # apply mask to each emission flux array, ivar, and the PSF
        ma_nz_halpha = np.ma.array(halpha, mask = m)
        ma_nz_hbeta = np.ma.array(hbeta, mask = m)
        ma_nz_SII_6718 = np.ma.array(SII_6718, mask = m)
        ma_nz_SII_6732 = np.ma.array(SII_6732, mask = m)
        ma_nz_SII =  ma_nz_SII_6718 + ma_nz_SII_6732
        ma_nz_OI = np.ma.array(OI, mask = m)
        ma_nz_OIII = np.ma.array(OIII, mask = m)
        ma_nz_NII = np.ma.array(NII, mask = m)
        
        ma_nz_SII_6718_ivar = np.ma.array(SII_6718_ivar, mask = m)
        ma_nz_SII_6732_ivar = np.ma.array(SII_6732_ivar, mask = m)
        ma_nz_SII_ivar = ma_nz_SII_6718_ivar + ma_nz_SII_6732_ivar
        ma_nz_OI_ivar = np.ma.array(OI_ivar, mask = m)
        ma_nz_OIII_ivar = np.ma.array(OIII_ivar, mask = m)
        ma_nz_NII_ivar = np.ma.array(NII_ivar, mask = m)
        
        # variance
        ma_nz_PSF = np.ma.array(PSF, mask = m)
        
        w = ma_nz_PSF / (ma_nz_PSF**2).sum()
        
        SII_variance = np.sum(w**2 / ma_nz_SII_ivar)
        OI_variance = np.sum(w**2 / ma_nz_OI_ivar)
        OIII_variance = np.sum(w**2 / ma_nz_OIII_ivar)
        NII_variance = np.sum(w**2 / ma_nz_NII_ivar)
        
        # standard deviation
        SII_sigma = np.sqrt(SII_variance)
        OI_sigma = np.sqrt(OI_variance)
        OIII_sigma = np.sqrt(OIII_variance)
        NII_sigma = np.sqrt(NII_variance)
        
        # central flux of each emission line
        halpha_cf = (ma_nz_halpha*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()
        hbeta_cf = (ma_nz_hbeta*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()
        SII_cf = (ma_nz_SII*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()
        OI_cf = (ma_nz_OI*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()
        OIII_cf = (ma_nz_OIII*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()
        NII_cf = (ma_nz_NII*ma_nz_PSF).sum() / (ma_nz_PSF**2).sum()

        log_halpha_cf = np.log10(halpha_cf)
        log_hbeta_cf = np.log10(hbeta_cf)
        log_SII_cf = np.log10(SII_cf)
        log_OI_cf= np.log10(OI_cf)
        log_OIII_cf= np.log10(OIII_cf)
        log_NII_cf = np.log10(NII_cf)
        
        log_SII_Ha_cf = np.log10(SII_cf/halpha_cf)
        log_OI_Ha_cf= np.log10(OI_cf/halpha_cf)
        log_OIII_Hb_cf= np.log10(OIII_cf/hbeta_cf)
        log_NII_Ha_cf = np.log10(NII_cf/halpha_cf)

        return i, halpha_cf, hbeta_cf, SII_cf, OI_cf, OIII_cf, NII_cf, log_halpha_cf, log_hbeta_cf, log_SII_cf, log_OI_cf, log_OIII_cf, log_NII_cf, log_SII_Ha_cf, log_OI_Ha_cf, log_OIII_Hb_cf, log_NII_Ha_cf, SII_variance, OI_variance, OIII_variance, NII_variance, SII_sigma, OI_sigma, OIII_sigma, NII_sigma

# Measure central flux for each good galaxy
starttime = time.time()
centralflux_data = []

for i in MNSA_plateifu[pipe3D_index]:
    centralflux_data.append(central_flux(i))
    
endtime = time.time()
elapsedtime = endtime - starttime
    
# make centralflux_data a structured ndarray
cf_dtype = ([('plateifu', np.compat.unicode, 15),('halpha_cf', np.float64), ('hbeta_cf', np.float64),
             ('SII_cf', np.float64), ('OI_cf', np.float64),('OIII_cf', np.float64),
             ('NII_cf', np.float64), ('log_halpha_cf', np.float64),('log_hbeta_cf', np.float64),
             ('log_SII_cf', np.float64), ('log_OI_cf', np.float64), ('log_OIII_cf', np.float64), 
             ('log_NII_cf', np.float64), ('log_SII_Ha_cf', np.float64), ('log_OI_Ha_cf', np.float64),
             ('log_OIII_Hb_cf', np.float64), ('log_NII_Ha_cf', np.float64), ('SII_variance', np.float64), 
             ('OI_variance', np.float64),('OIII_variance', np.float64), ('NII_variance', np.float64),
             ('SII_sigma', np.float64),('OI_sigma', np.float64),('OIII_sigma', np.float64),
             ('NII_sigma', np.float64)])

centralflux_data = np.array(centralflux_data, dtype = cf_dtype)

# save data to a FITS file

filename = 'centralflux.fits'
fitsio.write(filename, centralflux_data, clobber=True)
data = fitsio.FITS(filename)
print(data)
data.close()

# Kewley (2006) lines
# first create a linspace of points to plot the classification lines
x_SII_sf = np.linspace(-1.5,0.065)
x_SII_sy_liner = np.linspace(-0.31,1.5)

x_NII_sf = np.linspace(-1.31, 0.045)
x_NII_comp = np.linspace(-2.2, 0.35)

x_OI_sf = np.linspace(-2.5, -0.7)
x_OI_sy_liner = np.linspace(-1.12, 0.5)

def starformation_SII(log_SII_Ha):
    """Star formation classification line for log([SII]/Ha)"""
    return 0.72/(log_SII_Ha - 0.32) + 1.30

def seyfert_liner_SII(log_SII_Ha):
    """Seyfert and LINER classification line for log([SII]/Ha)"""
    return 1.89 * log_SII_Ha + 0.76

def seyfert_liner_OI(log_OI_Ha):
    """Seyfert and LINER classification line for log([OI]/Ha)"""
    return 1.18 * log_OI_Ha + 1.30

def starformation_OI(log_OI_Ha):
    """Star formation classification line for log([OI]/Ha)"""
    return 0.73 / (log_OI_Ha + 0.59) + 1.33

def composite_NII(log_NII_Ha):
    """Composite galaxy classification line for log([NII]/Ha)"""
    return 0.61/(log_NII_Ha - 0.47) + 1.19

def starformation_NII(log_NII_Ha):
    """Composite galaxy and LINER classification line for log([NII]/Ha)"""
    return 0.61 / (log_NII_Ha - 0.05) + 1.3

#log([OIII]/H-beta)/([SII]/H-alpha)
plt.figure(figsize = (5,5))
plt.scatter(centralflux_data['log_SII_Ha_cf'], centralflux_data['log_OIII_Hb_cf'],  s = 50, alpha = 0.3, marker ='.', linestyle = 'None')
plt.plot(x_SII_sf, starformation_SII(x_SII_sf), '-k')
plt.plot(x_SII_sy_liner, seyfert_liner_SII(x_SII_sy_liner), '--k')
plt.text(-1,1.2, 'Seyfert', fontsize = 12)
plt.text(-1.35,-0.25, 'Star Formation', fontsize = 12)
plt.text(0.4, 0.15, 'LINER', fontsize = 12)

plt.title('BPT Diagram')
plt.xlabel(r'log([SII]/H${\alpha}$)')
plt.ylabel(r'log([OIII]/H${\beta}$)')
plt.xlim(-1.5,1.0)
plt.ylim(-1.0,1.5)
plt.tight_layout()
plt.minorticks_on()
#plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_SII_Ha_OIII_Hb.png', overwrite = True)
plt.show()

#log([OIII]/H-beta) & ([OI]/H-alpha)
plt.figure(figsize = (5,5))
plt.scatter(centralflux_data['log_OI_Ha_cf'], centralflux_data['log_OIII_Hb_cf'],  s = 50, alpha = 0.3, marker ='.', linestyle = 'None')
plt.plot(x_OI_sf, starformation_OI(x_OI_sf), '-k')
plt.plot(x_OI_sy_liner, seyfert_liner_OI(x_OI_sy_liner), '--k')
plt.text(-1.5,1.05, 'Seyfert', fontsize = 12)
plt.text(-0.6, 0.15, 'LINER', fontsize = 12)

plt.title('BPT Diagram')
plt.xlabel(r'log([OI]/H${\alpha}$)')
plt.ylabel(r'log([OIII]/H${\beta}$)')
plt.xlim(-2.5,0.5)
plt.ylim(-1.0,1.5)
plt.tight_layout()
plt.minorticks_on()
#plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_OI_Ha_OIII_Hb.png', overwrite = True)
plt.show()

#log([OIII]/H-beta) & ([NII]/H-alpha)
plt.figure(figsize = (5,5))
plt.scatter(centralflux_data['log_NII_Ha_cf'], centralflux_data['log_OIII_Hb_cf'],  s = 50, alpha = 0.3, marker ='.', linestyle = 'None')
plt.plot(x_NII_sf, starformation_NII(x_NII_sf), '--k')
plt.plot(x_NII_comp, composite_NII(x_NII_comp), '-k')
plt.text(-0.75,1.15, 'AGN', fontsize = 12)
plt.text(-1.7,-0.15, 'Star Formation', fontsize = 12)
plt.text(-0.29, -0.45, 'Comp', fontsize = 12)

plt.title('BPT Diagram')
plt.xlabel(r'log([NII]/H${\alpha}$)')
plt.ylabel(r'log([OIII]/H${\beta}$)')
plt.xlim(-2.0,1.0)
plt.ylim(-1.0,1.5)
plt.tight_layout()
plt.minorticks_on()
#plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_NII_Ha_OIII_Hb.png', overwrite = True)
plt.show()

print('Elapsed time is:', time.strftime("%Hh%Mm%Ss", time.gmtime(elapsedtime)))

# Now let's check our central flux values with those calculated from Pipe3D

# galaxy properties we are interested in
pipe3D_log_SII_Ha_cf = pipe3D_galproperties['log_SII_Ha_cen'][:]
pipe3D_log_OI_Ha_cf = pipe3D_galproperties['log_OI_Ha_cen'][:]
pipe3D_log_OIII_Hb_cf  = pipe3D_galproperties['log_OIII_Hb_cen'][:]
pipe3D_log_NII_Ha_cf = pipe3D_galproperties['log_NII_Ha_cen'][:]

pipe3D_log_SII_Ha_cf_err = pipe3D_galproperties['e_log_SII_Ha_cen'][:]
pipe3D_log_OI_Ha_cf_err = pipe3D_galproperties['e_log_OI_Ha_cen'][:]
pipe3D_log_OIII_Hb_cf_err = pipe3D_galproperties['e_log_OIII_Hb_cen'][:]
pipe3D_log_NII_Ha_cf_err = pipe3D_galproperties['e_log_NII_Ha_cen'][:]

delta_1 = centralflux_data['log_SII_Ha_cf'] -  pipe3D_log_SII_Ha_cf
delta_2 = centralflux_data['log_OI_Ha_cf'] - pipe3D_log_OI_Ha_cf
delta_3 = centralflux_data['log_OIII_Hb_cf'] - pipe3D_log_OIII_Hb_cf
delta_4 = centralflux_data['log_NII_Ha_cf'] - pipe3D_log_NII_Ha_cf

err_1 = delta_1/centralflux_data['SII_sigma'] 
err_2 = delta_2/centralflux_data['OI_sigma'] 
err_3 = delta_3/centralflux_data['OIII_sigma'] 
err_4 = delta_4/centralflux_data['NII_sigma'] 

# plot of errors
plt.figure(figsize = (5,5))
plt.plot(centralflux_data['log_SII_cf'], err_1, 'o')

plt.title('Error in Central Flux of SII Doublet')
plt.xlabel(r'log([SII]')
plt.ylabel(r'${\Delta}$log([SII]/H${\alpha}$)')
plt.tight_layout()
plt.grid()
plt.minorticks_on()

plt.figure(figsize = (5,5))
plt.plot(centralflux_data['log_OI_cf'], err_2, 'o')

plt.title('Error in Central Flux of OI')
plt.xlabel(r'log([OI]')
plt.ylabel(r'${\Delta}$log([OI]/H${\alpha}$)')
plt.tight_layout()
plt.grid()
plt.minorticks_on()

plt.figure(figsize = (5,5))
plt.plot(centralflux_data['log_OIII_cf'], err_3, 'o')

plt.title('Error in Central Flux of OIII')
plt.xlabel(r'log([OIII]')
plt.ylabel(r'${\Delta}$log([OIII]/H${\beta}$)')
plt.tight_layout()
plt.grid()
plt.minorticks_on()

plt.figure(figsize = (5,5))
plt.plot(centralflux_data['log_NII_cf'], err_4, 'o')

plt.title('Error in Central Flux of NII')
plt.xlabel(r'log([NII]')
plt.ylabel(r'${\Delta}$log([NII]/H${\alpha}$)')
plt.tight_layout()
plt.grid()
plt.minorticks_on()

plt.show()
