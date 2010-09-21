"""
Create the figure for the basic example

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np
import scipy.io as sio
from plottools import plotall
import pylab as P

sys.path.append('../PythonSrc')
import evaluation
import masking
import imputation


if __name__ == '__main__':

    
    btchroma145 = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32k/Caroline_No/beach_boys+Pet_Sounds+13-Caroline_No.mp3.mat')['btchroma']
    mask,masked_cols = masking.random_col_mask(btchroma145,ncols=1,win=30)
    recon,lt = imputation.lintransform_cols(btchroma145,mask,masked_cols,win=1)
    pos1 = masked_cols[0] - 7
    pos2 = masked_cols[0] + 7
    im1 = btchroma145[:,pos1:pos2].copy()
    im2 = (btchroma145 * mask)[:,pos1:pos2].copy()
    im3 = recon[:,pos1:pos2].copy()
    # plot all this
    fig = P.figure()
    fig.subplots_adjust(hspace=0.4)
    blackbarsfun = lambda: P.gca().axvline(linewidth=2,color='0.',x=6.5) and P.gca().axvline(linewidth=2,color='0.',x=7.5)
    plotall([im1,im2,im3],subplot=(3,1),cmap='gray_r',
            title=['original','original masked','reconstruction'],
            axvlines=blackbarsfun,colorbar=False,xticks=[()]*3)
    P.show()
