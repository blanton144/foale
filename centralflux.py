#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import foale.mnsa
import numpy as np
import fitsio

# initialize data for all galaxies in survey
version = 'dr17-v0.1'
summary_file = os.path.join(os.getenv('MNSA_DATA'), version,'mn-{v}-summary.fits')
summary_file = summary_file.format(v=version)
f = fitsio.read(summary_file)

# extract plateifu for each galaxy
plate_IFU = f['plateifu']
plate = f['plate']

# removing these elements because FITSIO failed to find these files
plate_IFU = np.delete(plate_IFU, [22,23,24,25,26,27,28,29,30,31])

def central_flux(p_IFU):
    """Function to calculate central flux for each emission line of interest"""
    # Read in the cube's FITS file
    m = foale.mnsa.MNSA(plateifu = p_IFU, dr17 = True)
    m.read_cube()
    wave = m.cube['WAVE'].read()
    flux = m.cube['FLUX'].read()
    PSF = m.cube['RPSF'].read()

    # Read in the maps FITS file
    m.read_maps()

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

    # apply mask to each emission flux measurement array
    ma_nz_halpha = np.ma.array(halpha, mask = m)
    ma_nz_hbeta = np.ma.array(hbeta, mask = m)
    ma_nz_SII_6718 = np.ma.array(SII_6718, mask = m)
    ma_nz_SII_6732 = np.ma.array(SII_6732, mask = m)
    ma_nz_OI = np.ma.array(OI, mask = m)
    ma_nz_OIII = np.ma.array(OIII, mask = m)
    ma_nz_NII = np.ma.array(NII, mask = m)

    # array of the sum of the SII doublet
    ma_nz_SII =  ma_nz_SII_6718 + ma_nz_SII_6732


    # central flux of each emission line
    halpha_cf = (ma_nz_halpha*PSF).sum() / (PSF**2).sum()
    hbeta_cf = (ma_nz_hbeta*PSF).sum() / (PSF**2).sum()
    SII_cf = (ma_nz_SII*PSF).sum() / (PSF**2).sum()
    OI_cf = (ma_nz_OI*PSF).sum() / (PSF**2).sum()
    OIII_cf = (ma_nz_OIII*PSF).sum() / (PSF**2).sum()
    NII_cf = (ma_nz_NII*PSF).sum() / (PSF**2).sum()

    return[halpha_cf, hbeta_cf, SII_cf, OI_cf, OIII_cf, NII_cf]

# Measure central flux for each galaxy
#for i in plate_IFU:
#    central_flux(i)

#print(np.where(plate_IFU == '10141-6104'))
#print(plate)
print(np.where(plate == 10142))
