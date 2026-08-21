#!/usr/bin/env python

"""
PSF Analysis Tool
2026-06-08, nmosko@lowell.edu

Using astropy photutils, run source finder and then analyze PSFs across image

Instrument specific parameters are defined in associated improc_config.py

-------------------------------------------------------------

usage: psf_check.py [-h] [--impath IMPATH] [--imsuff IMSUFF]
                    [--boxsize BOXSIZE] [--plot]

Analyze PSF across images

options:
  -h, --help            show this help message and exit
  --impath IMPATH      Directory to scan for images, default = ./
  --imsuff IMSUFF      Image file suffix, default = solved.fits
  --boxsize BOXSIZE, -b BOXSIZE
                        Box size in pixels around each source to fit PSF, must
                        be odd integer, default = 15
  --plot, -p            Generate plot

-------------------------------------------------------------
 
"""

import argparse
import glob

import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.nddata import Cutout2D

from photutils.psf import PSFPhotometry, CircularGaussianPRF
from photutils.detection import DAOStarFinder
from photutils import psf

from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import improc_config

# supress warnings related to fit convergence
import warnings
warnings.filterwarnings(
    "ignore", message=r'.*One or more fit\(s\) may not have converged.*')
    
############################
# Generate plot of residuals
############################

def fwhm_plot(image,data,background,sources,boxsize):
        
    # construct and save plot
    
    # setup figure
    fig, ax = plt.subplots(figsize=(6,6),layout='constrained')
    fig.suptitle(image+r', PSF Analysis: $\overline{FWHM}$ = '+f'{np.mean(sources["fwhm"]):.2f}',size=11)

    # marker size scaled by 2 * normalized flux of sources
    norm_flux = 2*sources['flux'] / np.mean(sources['flux'])
    sym_size = [0 if f < 0 else f for f in norm_flux]

    # all sources used to fit PSF
    # account for 10% buffer around image edges with xoff and yoff
    xoff = data.shape[1] // 8
    yoff = data.shape[0] // 8
    ax.scatter(sources['x_centroid']+xoff,sources['y_centroid']+yoff, s=sym_size,marker='o',color='slateblue',label='N = '+str(len(sources))+' brightest sources')
    ax.legend(loc='upper left')

    # Set aspect and ticks of the main Axes.
    ax.set_aspect(1.)
    ax.tick_params(axis='both', direction='in')

    # main plot limits
    ax.set_xlim(0,data.shape[1]+2*xoff)
    ax.set_ylim(0,data.shape[0]+2*yoff)

    # create new Axes on the right and top of the current Axes for margin plots
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_psf = divider.append_axes("top", 1., pad=0.5)
    ax_fwhm = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
    
    # margin plots
    #
    # representative PSF fit from one star in list
    # 2D cutout of star
    if len(sources) <= 1:
        ind = 0    # first (brightest) star
    else:
        ind = int(len(sources)/2)   # middle star in list
        ind=0
    cutout = Cutout2D(data - background, (sources['x_centroid'][ind],sources['y_centroid'][ind]), boxsize)
    # if source is on the edge of frame, a full cutout is not possible
    # try the next star in the list until a full cutout is possible
    while cutout.shape != (boxsize,boxsize):
        ind +=1
        cutout = Cutout2D(data - background, (sources['x_centroid'][ind],sources['y_centroid'][ind]), boxsize)
    xcut, ycut = cutout.input_position_cutout   # star's position within the thumbnail
    
    # fit star
    psf_model = CircularGaussianPRF(fwhm=sources['fwhm'][ind])
    psfphot = PSFPhotometry(psf_model, fit_shape=boxsize)
    init_params = Table({'x': [xcut], 'y': [ycut], 'flux': [sources['flux'][ind]]})
    result = psfphot(cutout.data, init_params=init_params)
 
    # build a model instance carrying the fitted parameters
    row = result[0]
    fit_model = psf_model.copy()
    fit_model.x_0 = row['x_fit']
    fit_model.y_0 = row['y_fit']
    fit_model.flux = row['flux_fit']
    
    # radial profile: data
    yy, xx = np.mgrid[0:boxsize, 0:boxsize]
    r_pix = np.sqrt((xx - row['x_fit'])**2 + (yy - row['y_fit'])**2)

    # radial profile: model
    r = np.linspace(0, boxsize, 200)
    model_radial = fit_model(row['x_fit'] + r, row['y_fit'])

    # circle star in main plot
    ax.plot(sources['x_centroid'][ind]+xoff,sources['y_centroid'][ind]+yoff,marker='o',ms=8,mec='firebrick',mfc='none')

    # data and PSF model in margin plot
    ax_psf.tick_params(axis='both', direction='in')
    ax_psf.plot(r_pix.ravel(), cutout.data.ravel(), '.', color='gray', ms=3, label='data (pixels)')
    ax_psf.plot(r, model_radial, '-', label='PSF fit',c='firebrick',lw=1)
    ax_psf.set_xlabel('radius (pix)')
    ax_psf.set_ylabel('counts')

    # thumnail inset of selected star
    ax_thumb = fig.add_axes([0.765, 0.766, 0.18, 0.18])
    ax_thumb.imshow(cutout.data, cmap="gray_r", origin="lower", norm='log')
    aperture = plt.Circle((xcut, ycut), sources['fwhm'][ind],fill=False,ec='firebrick')
    ax_thumb.add_patch(aperture)
    ax_thumb.set_xticks([])
    ax_thumb.set_yticks([])

    # margin plot of PSF FWHM across y axis
    ax_fwhm.yaxis.set_tick_params(labelleft=False)
    ax_fwhm.set_xlabel('FWHM')
    ax_fwhm.scatter(sources['fwhm'],sources['y_centroid'],color='k',s=1,label='PSF Roundness, y-axis')

    ax_fwhm.yaxis.set_label_position("right")
    ax_fwhm.yaxis.tick_right()
    ax_fwhm.tick_params(axis='both', direction='in')

    # annotations
    ax.set_xlabel('X (pix)')
    ax.set_ylabel('Y (pix)')

    # save figure
    plt.savefig(image.replace('.fits','_psf.png'),dpi=150)
    plt.close(fig)
    plt.clf()
    
    return
    
###############################
# Measure PSF FWHM across image
###############################

def find_fwhm(image,params):

    # retrieve header keywords for processing and summary file
    date_obs = fits.getval(image, 'DATE-OBS', ext=0)     # UT date of observation
    filt = fits.getval(image, 'FILTER', ext=0)          # Filter
    obj = fits.getval(image, 'OBJECT', ext=0)           # Object
    ra_header = fits.getval(image, 'RA', ext=0)    # right ascension
    dec_header = fits.getval(image, 'DEC', ext=0)  # declination
    pix_scale = params['pix_scale']         # unbinned pixel scale
    max_counts = params['max_counts']         # max counts before non-linear

    # convert RA/DEC from Deg to H:M:S if needed
    if params['radec_units']:
        coord = SkyCoord(ra=ra_header * u.degree, dec=dec_header * u.degree, frame='icrs')
        ra = coord.ra.to_string(unit=u.hourangle, sep=':', pad=True, precision=2)
        dec = coord.dec.to_string(unit=u.degree, sep=':', pad=True, alwayssign=True, precision=2)
    else:
        ra, dec = ra_header, dec_header
   
    # read fits data
    hdu = fits.open(image)
    
    # define 10% buffer around edge to ignore sources
    x0 = hdu[0].data.shape[1] // 10
    x1 = 9 * hdu[0].data.shape[1] // 10
    y0 = hdu[0].data.shape[0] // 10
    y1 = 9 * hdu[0].data.shape[0] // 10
    data = hdu[0].data[y0:y1,x0:x1]

    # derive background, 3 sigma clipped
    sig = 3.0
    if simpleback:
        # simplest is single value, sigma clipped median, from astropy.stats
        mean, median, std = sigma_clipped_stats(data, sigma=sig)
        background = median
    else:
       # 2D background maps with photutils.background
        from photutils.background import Background2D, MedianBackground

        sigma_clip = SigmaClip(sigma=sig)
        bkg_estimator = MedianBackground()
        bkg = Background2D(data, boxsize, filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        background = bkg.background
        std = bkg.background_rms_median

    # extract sources from background subtracted image
    threshold = 5.0 * std   # source detection threshold
    daofind = DAOStarFinder(threshold, fwhm=2.5,n_brightest=100,exclude_border=True,peak_max=max_counts)
    sources = daofind(data-background) # astropy Table

    # list of (x, y) coordinates of sources
    xypos = zip(sources['x_centroid'],sources['y_centroid'])

    # measure FWHM across sources within a square of size boxsize
    fwhm = psf.fit_fwhm(data-background,xypos=xypos,fit_shape=boxsize)

    # add fwhm to sources table and write results to file
    sources['fwhm'] = fwhm
    sources.write(image.replace('.fits','_psf.dat'), format='ascii.fixed_width_two_line', overwrite=True)

    # summary data
    psf_dat = (image, date_obs, ra, dec, filt, obj, len(sources), np.mean(fwhm), np.std(fwhm), np.mean(sources['roundness2']), np.mean(sources['roundness1']))
             
    # create plot of FWHM across image
    if do_plot:
        fwhm_plot(image,data,background,sources,boxsize)

    return psf_dat
        
##############################
# Main Block
##############################
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Analyze PSF across images')
    parser.add_argument('--impath',  help='Directory to scan for images, default = ./', default='./')
    parser.add_argument('--imsuff',  help='Image file suffix, default = solved.fits', default='solved.fits')
    parser.add_argument('--boxsize', '-b', help='Box size in pixels around each source to fit PSF, must be odd integer, default = 15', default=15)
    parser.add_argument('--plot', '-p', action='store_true', help='Generate plot')
    parser.add_argument('--simpleback', '-s', action='store_true', help='Use simple single value median background')

    args = parser.parse_args()
    
    img_path = args.impath
    # add trailing slash / if missing from image path name
    if img_path[-1] != '/':
        img_path += '/'
    imsuff = args.imsuff
    boxsize = args.boxsize
    do_plot = args.plot
    simpleback = args.simpleback

    print('-'*80)
    print('RUN PSF CHECK\n')

    # summary table
    # diag_round = roundness1 = SROUND = comparison of diagonal quadrants around centroid
    # xy_round = roundness2 = GROUND = normalized height difference in x and y gaussian fits
    #   = 0 = perfectly round
    #   < 0 = extended along x axis
    #   > 0 = extended along y axis
    summary = Table(names=('image', 'UT Date','RA', 'Dec', 'filter', 'object', 'N_sources', 'fwhm_mean', 'fwhm_stdev', 'xy_round_mean', 'diag_round_mean'), dtype=(str,str,str,str,str,str,int,float,float,float,float))

    # retrieve list of images to analyze
    # recursively scan through all sub-directories relative to img_path
    images = glob.glob(img_path+'**/*'+imsuff, recursive=True)
    print('Processing '+str(len(images))+' images...\n')

    # loop through images and measure PSF FWHMs
    if len(images) > 0:
    
        # pull telescope+instrument header keywords from first image
        print('Checking image ' + images[0] + ' for configuration parameters')
        telescope = fits.getval(images[0], 'TELESCOP', ext=0)
        instrument = fits.getval(images[0], 'INSTRUME', ext=0)
        tel_inst = telescope + '_' + instrument
        print('   From image header: TELESCOP = ' + telescope)
        print('   From image header: INSTRUME = ' + instrument + '\n')

        # use tel_inst to retrieve configuration params in improc_config.py
        params = improc_config.parameters(tel_inst)
    
        # print statement on background substraction method
        if simpleback:
            print('Background subtraction: Using single median value\n')
        else:
            print('Background subtraction: Using 2D map based on a grid of median values\n')

        # process images
        for im in images:
        
            # call main function for measuring PSF FWHM across image
            summary_row = find_fwhm(im,params)
            
            # add image data to summary table
            summary.add_row(summary_row)
            
            print(im,' Mean FWHM +/- stdev (pix) =',f"{summary[-1]['fwhm_mean']:.3f}",'+/-',f"{summary[-1]['fwhm_stdev']:.3f}")
                        
        # write all summary data to file
        summary.write('psf_summary.txt', format='ascii.fixed_width_two_line', formats={'fwhm_mean':'0.3f', 'xy_round_mean':'0.3f', 'diag_round_mean':'0.3f'}, overwrite=True)
        
        # FWHM across all images
        mean_fwhm = np.mean(summary['fwhm_mean'])
        std_fwhm = np.std(summary['fwhm_mean'])
        
        # print some information to terminal
        print('\nPSF SUMMARY')
        print(str(len(summary)) + ' files in path ' + img_path)
        print('Mean FWHM (pixels) = ' + f'{mean_fwhm:.3f}')
        print('Std. dev. FWHM (pixels) = ' + f'{std_fwhm:.3f}')
        print('\nSummary file saved to psf_summary.txt')
        print('\nPSF CHECK DONE')
        print('-'*80)

    else:
        print('No *solved.fits images found in path '+img_path)
