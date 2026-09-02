#!/usr/bin/env python

"""
Image Processing Configuation Script
2026-05-14, nmosko@lowell.edu
"""

import sys

######################################
# telescope/instrument configurations
#
implemented_instruments = ['Lowell Hall 1.1-m f_E2V CCD-231 4096x4112', 'Lowell Hall 1.1-m f/8_E2V CCD-231 4096x4112', 'DCT_lmi', 'PJ1M_PJ1m_Moravian']


# Hall 42", NASA42
hall_nasa42 = {
    'binning': ('ADELX_01', 'ADELY_01'),    # x and y bin factors
    'search_radius': 0.5,           # query catalog within radius of field center [deg]
    'pix_scale': 0.368,              # unbinned pixel scale [arcsec]
    'radec_units': False,             # boolean to convert header RA/Dec deg to H:M:S
    'poly_n': 3,                     # polynomial order of SIP WCS correction
    'max_counts': 45000             # max counts before non-linear
}

# LDT, LMI
ldt_lmi = {
    'binning': ('CCDSUM#blank0', 'CCDSUM#blank1'),    # x and y bin factors
    'search_radius': 0.1,           # query catalog within radius of field center [deg]
    'pix_scale': 0.12,               # unbinned pixel scale [arcsec]
    'radec_units': False,             # boolean to convert header RA/Dec deg to H:M:S
    'poly_n': 3,                     # polynomial order of SIP WCS correction
    'max_counts': 50000             # max counts before non-linear
}

# PJ1m, Moravian
pj1m_moravian = {
    'binning': ('XBINNING', 'YBINNING'),    # x and y bin factors
    'search_radius': 0.5,           # query catalog within radius of field center [deg]
    'pix_scale': 0.13,               # unbinned pixel scale [arcsec]
    'radec_units': True,             # boolean to convert header RA/Dec deg to H:M:S
    'poly_n': 3,                     # polynomial order of SIP WCS correction
    'max_counts': 50000             # max counts before non-linear
}

##########################################################################
# translate TELESCOP+'_'+INSTRUME keywords into parameter set defined here
#
params = {
    'Lowell Hall 1.1-m f_E2V CCD-231 4096x4112':    hall_nasa42,
    'Lowell Hall 1.1-m f/8_E2V CCD-231 4096x4112':  hall_nasa42,
    'DCT_lmi':                                      ldt_lmi,
    'PJ1M_PJ1m_Moravian':                           pj1m_moravian
}

######################
# retrieve parameters
#
def parameters(tel_inst):
    """ Retrieve telescope specific parameter set
    """

    if tel_inst not in implemented_instruments:
        print(tel_inst+' not implemented. Exiting...')
        exit()
    
    tel_param = params[tel_inst]
    
    return tel_param

#############
# Main block
#
if __name__ == '__main__':
    
    tel_name = sys.argv[1]
    inst_name = sys.argv[2]

    tel_inst = tel_name+'_'+inst_name

    parameters(tel_inst)
