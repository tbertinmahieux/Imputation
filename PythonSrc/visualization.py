"""
A set of visualization / demo tools about imputation, mainly to
compare different techniques for a same imputation task.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np
# imputation tools
import imputation as IMPUTATION
import imputation_plca as IMPUTATION_PLCA
import evaluation as EVAL
# Ron's plotting tool
import pylab as plt
from plottools import plotall


def compare_all(btchroma,mask,masked_cols,codebook=None):
    """
    Compare all the algorithms we have so far.
    A lot of parameters hard-coded, but...
    It will get improved.
    We display:
      1) original beat chromagram
      2) masked beat chromagram
      3) imputation by random
      4) imputation by averaging nearby columns
      5) imputation by knn on the rest of the song
      6) imputation by linear prediction
      7) imputation by NMF
      8) imputation by codebook, if provided

    To init mask and masked_cols, something like:
    mask,masked_cols = masking.random_col_mask(btchroma,ncols=1,win=30)
    """
    # for displaying purposes, display window size 50
    pos1 = masked_cols[0] - 5
    pos2 = masked_cols[0] + 5
    allimages = []
    titles = []
    xlabels = []
    # original beat chroma and masked one
    im1 = btchroma[:,pos1:pos2].copy()
    im2 = (btchroma * mask)[:,pos1:pos2].copy()
    allimages.append(im1)
    titles.append('original beat chroma')
    xlabels.append('')
    allimages.append(im2)
    titles.append('masked beat chroma')
    xlabels.append('')
    # 3) random
    recon = IMPUTATION.random_col(btchroma,mask,masked_cols)
    im3_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im3 = recon[:,pos1:pos2].copy()
    allimages.append(im3)
    titles.append('random')
    xlabels.append('err='+str(im3_err))
    # 4) average nearby columns
    recon = IMPUTATION.average_col(btchroma,mask,masked_cols,win=3)
    im4_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im4 = recon[:,pos1:pos2].copy()
    allimages.append(im4)
    titles.append('average 2 nearby cols')
    xlabels.append('err='+str(im4_err))
    # 5) knn
    recon,used_cols = IMPUTATION.eucldist_cols(btchroma,mask,masked_cols,win=7)
    im5_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im5 = recon[:,pos1:pos2].copy()
    allimages.append(im5)
    titles.append('knn for the whole song')
    xlabels.append('err='+str(im5_err))
    # 6) linear prediction
    recon,proj = IMPUTATION.lintransform_cols(btchroma,mask,masked_cols,win=1)
    im6_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im6 = recon[:,pos1:pos2].copy()
    allimages.append(im6)
    titles.append('linear prediction (1 col)')
    xlabels.append('err='+str(im6_err))
    # 7) SIPLCA
    res = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                              4,mask,win=5)
    W, Z, H, norm, recon, logprob = res
    im7_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im7 = recon[:,pos1:pos2].copy()
    allimages.append(im7)
    titles.append('SIPLCA,rank=4,win=5')
    xlabels.append('err='+str(im7_err))
    # 7) SIPLCA 2
    res = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                              25,mask,win=10)
    W, Z, H, norm, recon, logprob = res
    im7_err_bis = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
    im7_bis = recon[:,pos1:pos2].copy()
    allimages.append(im7_bis)
    titles.append('SIPLCA,rank=25,win=10')
    xlabels.append('err='+str(im7_err_bis))
    # 8) codebook
    if codebook != None:
        cb = [p.reshape(12,p.size/12) for p in codebook]
        recon,used_codes = IMPUTATION.codebook_cols(btchroma,mask,masked_cols,cb)
        im8_err = EVAL.recon_error(btchroma,mask,recon,measure='eucl')
        im8 = recon[:,pos1:pos2].copy()
        allimages.append(im8)
        titles.append('codebook, '+str(codebook.shape[0])+' codes of length '+str(codebook.shape[1]/12))
        xlabels.append('err='+str(im8_err))
    # ALL IMAGES CREATED
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    blackbarsfun = lambda: plt.gca().axvline(linewidth=2,color='0.',x=4.5) and plt.gca().axvline(linewidth=2,color='0.',x=5.5)
    # plotall
    plotall(allimages,subplot=(3,3),cmap='gray_r',title=titles,xlabel=xlabels,axvlines=blackbarsfun,colorbar=False)


