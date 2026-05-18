#!/usr/bin/env python

"""
Astrometry Solver
2026-05-14, nmosko@lowell.edu

Uses local installation of astrometry.net to derive astrometric solutions

Instrument specific parameters are defined in associated astrom_config.py

This script will create new versions of images with a _solved.fits suffix

-------------------------------------------------------------

usage: astrom_solve.py [-h] [--imgpath IMGPATH] [--cleanup] [--plot]

Astrometrically solve images.

optional arguments:
  -h, --help         show this help message and exit
  --imgpath IMGPATH  Directory to scan for images
  --cleanup, -c      Delete output files
  --plot, -p         Generate residual plot

-------------------------------------------------------------

Some brief setup instructions for installing astrometry.net:

$ conda install -c conda-forge astrometry
$ conda install conda-forge::fitsio

Index files must be downloaded to anaconda environment, e.g.:
~/anaconda3/envs/pp/data

Index files are available here:
https://data.astrometry.net/

5200 series, LIGHT version seems to work reasonably well with typical images
Shell script with the following form can be used to download all of the index files:

# index-5000-*
for ((i=0; i<48; i++)); do
    I=$(printf %02i $i)
    wget https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/index-5200-$I.fits
done

Save and run this shell script from ~/anaconda3/envs/pp/data
$ sh ./getindex.sh
 
"""

import argparse
import glob
import os
import warnings

import numpy as np

from astropy.utils.exceptions import AstropyWarning
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt

import astrom_config

############################
# Generate plot of residuals
############################

def residual_plot(image):

    # all sources extracted from the image
    img_sources = Table.read(image.replace('.fits','.xy'))
    
    # image sources matched to catalog
    # warnings wrap avoids non-standard units warning message
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=AstropyWarning)
        cat_sources = Table.read(image.replace('.fits','.corr'))
    
    # construct and save plot
    #
    # symbol size scaled by mean FLUX of detected sources
    norm_flux = img_sources['FLUX'] / np.mean(img_sources['FLUX'])
    sym_size = [0 if f < 0 else f for f in norm_flux]
    
    # all detected sources
    plt.scatter(img_sources['X'],img_sources['Y'],s=sym_size,color='slateblue',label='Extracted sources')
    
    # catalog sources
    plt.scatter(cat_sources['field_x'],cat_sources['field_y'], s=max(sym_size),marker='o',ec='grey',facecolors='none',label='Catalog stars')
    
    # residuals on astrometric solution
    dx = (cat_sources['field_x'] - cat_sources['index_x'])*1
    dy = (cat_sources['field_y'] - cat_sources['index_y'])*1
    mean_residual = np.mean(np.sqrt(np.array(dx)**2 + np.array(dy)**2))
    max_residual = np.sqrt(sorted(dx)[-1]**2 + sorted(dy)[-1]**2)
    
    # add residual vectors
    q = plt.quiver(cat_sources['field_x'],cat_sources['field_y'], dx,dy,width=0.002,headwidth=0,color='grey')

    # annotations
    plt.xlabel('X (pix)')
    plt.ylabel('Y (pix)')
    plt.title(image+ '\n Residuals (N='+str(len(cat_sources))+'): max = '+f'{max_residual:.3f} pix, mean = ' + f'{mean_residual:.3f} pix')
    plt.legend(loc='lower right',ncols=2)
    plt.tight_layout()
    plt.savefig(image.replace('.fits','_solved.png'),dpi=150)
    plt.clf()
    
##############################
# Astrometrically solve images
##############################

def astrom_solve(image,params):

    # retrieve header keywords
    ra = fits.getval(image, 'RA', ext=0)    # right ascension
    dec = fits.getval(image, 'DEC', ext=0)  # declination
    pix_scale = params['pix_scale']         # unbinned pixel scale
   
    # on-chip bin factor
    if isinstance(params['binning'][0],int):
        # compute binned pixel scale
        bin_scale = pix_scale * params['binning'][0]
    elif isinstance(params['binning'][0],str):
        # compute binned pixel scale
        bin_scale = pix_scale * fits.getval(image, params['binning'][0], ext=0)

    # upper and lower limits on binned pixel scale, 90% and 110% of pixel scale
    scale_low = f'{bin_scale * 0.9:.3f}'
    scale_high = f'{bin_scale * 1.1:.3f}'
    
    # construct call to astrometry.net solve-field function
    # parameters:
    #   -scale-low, scale-high      lower and upper bounds to the pixel scale
    #   -O              overwrite previous results
    #   -ra, -dec       ra,dec from header
    #   --radius        query catalog within this radius of field center [deg]
    #   --no-plots      supress output plots
    #   -t              polynomial order of SIP WCS correction
    #   --no-verify     dont try to verify existing WCS
    #   -k              file name for full list of xy positions of extracted sources
    #   -S              solved file, indicates image was solved
    #   -M              match file, quad match that solved image
    #   -R              rdls file, extracted RA/DEC of extracted sources
    #   --temp-axy      axy file, augmented xy list
    #   -U              xyls file, pixel locations of reference stars
    #   -N              new fits file output
    #   -W              wcs file
    #
    solve_field = 'solve-field "' + image + '" --scale-low ' + str(scale_low) + ' --scale-high ' + str(scale_high) + ' --scale-units arcsecperpix -O --ra ' + ra + ' --dec ' + dec + ' --radius ' + str(params['search_radius']) + ' --no-plots -t ' + str(params['poly_n']) + ' --no-verify -k %s.xy -S none -M none -R none --temp-axy -U none -N %s_solved.fits'

    # run solve-field
    os.system(solve_field)
        
    # create plot of residuals
    if do_plot:
        residual_plot(image)

    # clean up output files from solve-field
    if cleanup:
        os.remove(image.replace('.fits','.corr'))
        os.remove(image.replace('.fits','.wcs'))
        os.remove(image.replace('.fits','.xy'))
        
##############################
# Main Block
##############################
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Astrometrically solve images.')
    parser.add_argument('--imgpath',  help='Directory to scan for images', default='./')
    parser.add_argument('--cleanup', '-c', action='store_true', help='Delete output files')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate residual plot')

    args = parser.parse_args()
    
    img_path = args.imgpath
    cleanup = args.cleanup
    do_plot = args.plot


    # retrieve list of images, exclude previously solved fits files
    # assumes original file name does not include the string 'solved'
    # recursively scan through all sub-directories relative to img_path
    all_fits = glob.glob(img_path+'**/*.fits', recursive=True)
    images = [im for im in all_fits if 'solved' not in im]

    # loop through images and solve astrometrically
    if len(images) > 0:
    
        # pull telescope+instrument header keywords from first image
        telescope = fits.getval(images[0], 'TELESCOP', ext=0)
        instrument = fits.getval(images[0], 'INSTRUME', ext=0)
        tel_inst = telescope + '_' + instrument

        # use tel_inst to retrieve configuration params in astrom_config.py
        params = astrom_config.parameters(tel_inst)
    
        # solve images
        for im in images:
            astrom_solve(im,params)
            
    else:
        print('No .fits images found in path '+img_path)
