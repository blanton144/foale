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

# extract plateifu for each galaxy
plate_IFU = f['plateifu']

# example slice
p = plate_IFU[0:20]

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

        # calculate central flux line ratios for BPT diagrams
        log_SII_Ha_cf = np.log10(SII_cf/halpha_cf)
        log_OI_Ha_cf= np.log10(OI_cf/halpha_cf)
        log_OIII_Hb_cf= np.log10(OIII_cf/hbeta_cf)
        log_NII_Ha_cf = np.log10(NII_cf/halpha_cf)

        return i, halpha_cf, hbeta_cf, SII_cf, OI_cf, OIII_cf, NII_cf, log_SII_Ha_cf, log_OI_Ha_cf, log_OIII_Hb_cf, log_NII_Ha_cf

# Measure central flux for each good galaxy

starttime = time.time()
centralflux_data = []

#for i in plate_IFU:
#    centralflux_data.append(central_flux(i))

# example slice to check before running all galaxies
for i in p:
    centralflux_data.append(central_flux(i))
    
endtime = time.time()
elapsedtime = endtime - starttime

# make centralflux_data a structured ndarray
cf_dtype = ([('plateifu', np.compat.unicode, 15),('halpha_cf', np.float64), ('hbeta_cf', np.float64),
             ('SII_cf', np.float64), ('OI_cf', np.float64),('OIII_cf', np.float64),
             ('NII_cf', np.float64), ('log_SII_Ha_cf', np.float64), ('log_OI_Ha_cf', np.float64),
             ('log_OIII_Hb_cf', np.float64),('log_NII_Ha_cf', np.float64)])

centralflux_data = np.array(centralflux_data, dtype = cf_dtype)

# save data to a FITS file

#filename = 'centralflux.fits'
#fitsio.write(filename, centralflux_data, clobber=True)
#data = fitsio.FITS(filename)
#print(data)
#data.close()

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

# plot diagrams

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
plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_SII_Ha_OIII_Hb.png', overwrite = True)
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
plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_OI_Ha_OIII_Hb.png', overwrite = True)
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
plt.savefig('/uufs/chpc.utah.edu/common/home/u6044257/Desktop/BPT_NII_Ha_OIII_Hb.png', overwrite = True)
plt.show()

print('Elapsed time is:', time.strftime("%Hh%Mm%Ss", time.gmtime(elapsedtime)))

# now let's check our central flux values with those calculated from Pipe3D
# here is an example slice before we do this for all galaxies
pipe3D = fitsio.FITS('SDSS17Pipe3D_v3_1_1.fits')
pipe3D_galproperties = pipe3D[1][0:20]

pipe3D_plateifu = pipe3D_galproperties['plateifu']
pipe3D_log_SII_Ha_cf = pipe3D_galproperties['log_SII_Ha_cen']
pipe3D_log_OI_Ha_cf = pipe3D_galproperties['log_OI_Ha_cen']
pipe3D_log_OIII_Hb_cf= pipe3D_galproperties['log_OIII_Hb_cen']
pipe3D_log_NII_Ha_cf = pipe3D_galproperties['log_NII_Ha_cen']

pipe3D_log_SII_Ha_cf_err = pipe3D_galproperties['e_log_SII_Ha_cen']
pipe3D_log_OI_Ha_cf_err = pipe3D_galproperties['e_log_OI_Ha_cen']
pipe3D_log_OIII_Hb_cf_err = pipe3D_galproperties['e_log_OIII_Hb_cen']
pipe3D_log_NII_Ha_cf_err = pipe3D_galproperties['e_log_NII_Ha_cen']

# percent error between calculated log values and Pipe3D log values of central flux (in percent)
err_1 = np.abs((centralflux_data['log_SII_Ha_cf'] - pipe3D_log_SII_Ha_cf) / pipe3D_log_SII_Ha_cf) * 100 
err_2 = np.abs((centralflux_data['log_OI_Ha_cf'] - pipe3D_log_OI_Ha_cf) / pipe3D_log_OI_Ha_cf) * 100
err_3 = np.abs((centralflux_data['log_OIII_Hb_cf'] - pipe3D_log_OIII_Hb_cf) / pipe3D_log_OIII_Hb_cf) * 100
err_4 = np.abs((centralflux_data['log_NII_Ha_cf'] - pipe3D_log_NII_Ha_cf) / pipe3D_log_NII_Ha_cf) * 100
      
print('log_SII_Ha_cf error:',  err_1)
print('log_OI_Ha_cf error:', err_2)
print('log_OIII_Hb_cf error:', err_3)
print('log_NII_Ha_cf error:', err_4)

# the problem when running all galaxies is that centralflux_data has a shape of 11273 and pipe3D has a shape of 10220

"""
# now let's check our central flux values with those calculated from Pipe3D
pipe3D = fitsio.FITS('SDSS17Pipe3D_v3_1_1.fits')
pipe3D_galproperties = pipe3D[1].read()

pipe3D_plateifu = pipe3D_galproperties['plateifu']
pipe3D_log_SII_Ha_cf = pipe3D_galproperties['log_SII_Ha_cen']
pipe3D_log_OI_Ha_cf = pipe3D_galproperties['log_OI_Ha_cen']
pipe3D_log_OIII_Hb_cf= pipe3D_galproperties['log_OIII_Hb_cen']
pipe3D_log_NII_Ha_cf = pipe3D_galproperties['log_NII_Ha_cen']

pipe3D_log_SII_Ha_cf_err = pipe3D_galproperties['e_log_SII_Ha_cen']
pipe3D_log_OI_Ha_cf_err = pipe3D_galproperties['e_log_OI_Ha_cen']
pipe3D_log_OIII_Hb_cf_err = pipe3D_galproperties['e_log_OIII_Hb_cen']
pipe3D_log_NII_Ha_cf_err = pipe3D_galproperties['e_log_NII_Ha_cen']

# percent error between calculated log values and Pipe3D log values of central flux (in percent)
err_1 = np.abs((centralflux_data['log_SII_Ha_cf'] - pipe3D_log_SII_Ha_cf) / pipe3D_log_SII_Ha_cf) * 100 
err_2 = np.abs((centralflux_data['log_OI_Ha_cf'] - pipe3D_log_OI_Ha_cf) / pipe3D_log_OI_Ha_cf) * 100
err_3 = np.abs((centralflux_data['log_OIII_Hb_cf'] - pipe3D_log_OIII_Hb_cf) / pipe3D_log_OIII_Hb_cf) * 100
err_4 = np.abs((centralflux_data['log_NII_Ha_cf'] - pipe3D_log_NII_Ha_cf) / pipe3D_log_NII_Ha_cf) * 100

print(err_1, err_2, err_3, err_4)
"""
