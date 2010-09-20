"""
Set of functions to evalutate the masking and reconstruction procedures

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import glob
import numpy as np
import scipy.io as sio

import imputation as IMPUTATION
import imputation_plca as IMPUTATION_PLCA
import masking as MASKING

EPS = np.finfo(np.float).eps


def euclidean_dist_sq(v1,v2):
    """
    Trivial averaged squared euclidean distance from too flatten vectors
    """
    return np.square(v1.flatten()-v2.flatten()).mean()

def symm_kl_div(v1,v2):
    """
    Normalize and return the symmetric kldivergence
    """
    v1 = v1.copy() / v1.sum()
    v2 = v2.copy() / v2.sum()
    div1 = (v1 * np.log(v1 / v2 + EPS)).sum()
    div2 = (v2 * np.log(v2 / v1 + EPS)).sum()
    return (div1 + div2) / 2.


def recon_error(btchroma,mask,recon,measure='eucl'):
    """

    INPUT
       btchroma   - original feature matrix
       mask       - binary mask, same shape as btchroma
       recon      - reconstruction, same shape as btchroma
       measure    - 'eucl' (euclidean distance, default)
                  - 'kl' (symmetric KL-divergence)
    RETURN
       div        - divergence, or reconstruction error
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'bad mask shape'
    assert btchroma.shape == recon.shape,'bad recon shape'
    # get measure function
    if measure == 'eucl':
        measfun = euclidean_dist_sq
    elif measure == 'kl':
        measfun = symm_kl_div
    else:
        raise ValueError('wrong measure name, want eucl or kl?')
    # measure and done
    maskzeros = np.where(mask==0)
    return measfun( btchroma[maskzeros] , recon[maskzeros] )


def get_all_matfiles(basedir) :
    """
    From a root directory, go through all subdirectories
    and find all matlab files. Return them in a list.
    """
    allfiles = []
    for root, dirs, files in os.walk(basedir):
        matfiles = glob.glob(os.path.join(root,'*.mat'))
        for f in matfiles :
            allfiles.append( os.path.abspath(f) )
    return allfiles



def test_maskedcol_on_dataset(datasetdir,method='random',ncols=1,win=3,rank=4,codebook=None):
    """
    General method to test a method on a whole dataset for one masked column
    Methods are:
      - random
      - randomfromsong
      - average
      - codebook
      - knn_eucl
      - lintrans
      - siplca
    Used arguments vary based on the method
    """
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs_eucl = []
    errs_kl = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        ########## ALGORITHM DEPENDENT
        if method == 'random':
            recon = IMPUTATION.random_col(btchroma,mask,masked_cols)
        elif method == 'randomfromsong':
            recon = IMPUTATION.random_col_from_song(btchroma,mask,masked_cols)
        elif method == 'average':
            recon = IMPUTATION.average_col(btchroma,mask,masked_cols,win=win)
        elif method == 'codebook':
            if not type(codebook) == type([]):
                codebook = [p.reshape(12,codebook.shape[1]/12 for p in codebook]
            recon,used_codes = IMPUTATION.codebook_cols(btchroma,mask,masked_cols,codebook)
        elif method == 'knn_eucl':
                recon,used_cols = IMPUTATION.eucldist_cols(btchroma,mask,masked_cols,win=win)
        elif method == 'lintrans':
            recon,proj = IMPUTATION.lintransform_cols(btchroma,mask,masked_cols,win=win)
        elif method == 'siplca':
            res = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                                      rank,mask,win=win)
            W, Z, H, norm, recon, logprob = res
        else:
            print 'unknown method:',method
            return
        ########## ALGORITHM DEPENDENT END
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs_eucl.append( err )
        err = recon_error(btchroma,mask,recon,measure='kl')
        errs_kl.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs_eucl),'(',np.std(errs_eucl),')'
    print 'average kl divergence:',np.mean(errs_kl),'(',np.std(errs_kl),')'


    

def test_average_col_on_dataset(datasetdir,ncols=1,win=3):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs_eucl = []
    errs_kl = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon = IMPUTATION.average_col(btchroma,mask,masked_cols,win=win)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs_eucl.append( err )
        err = recon_error(btchroma,mask,recon,measure='kl')
        errs_kl.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs_eucl),'(',np.std(errs_eucl),')'
    print 'average kl divergence:',np.mean(errs_kl),'(',np.std(errs_kl),')'


def test_random_col_on_dataset(datasetdir,ncols=1):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH=50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs_eucl = []
    errs_kl = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon = IMPUTATION.random_col(btchroma,mask,masked_cols)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs_eucl.append( err )
        err = recon_error(btchroma,mask,recon,measure='kl')
        errs_kl.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs_eucl),'(',np.std(errs_eucl),')'
    print 'average kl divergence:',np.mean(errs_kl),'(',np.std(errs_kl),')'


def test_random_col_from_song_on_dataset(datasetdir,ncols=1):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH=50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon = IMPUTATION.random_col_from_song(btchroma,mask,masked_cols)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'
    

def test_codebook_cols_on_dataset(datasetdir,codebook,ncols=1):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      codebook   - array of patches 12xWIN
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    win = codebook[0].shape[1]
    total_cnt = 0
    errs = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon,used_codes = IMPUTATION.codebook_cols(btchroma,mask,masked_cols,codebook)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'



def test_eucldist_cols_on_dataset(datasetdir,ncols=1,win=3):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon,used_cols = IMPUTATION.eucldist_cols(btchroma,mask,masked_cols,win=win)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'


def test_lintransform_cols_on_dataset(datasetdir,ncols=1,win=1):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs = []
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        recon,proj = IMPUTATION.lintransform_cols(btchroma,mask,masked_cols,win=win)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'



def test_siplca_cols_on_dataset(datasetdir,ncols=1,rank=4,win=5):
    """
    Test the method of imputation by similar patterns in the same
    song on every mat file in a given dataset
    INPUT
      dataset    - dir we'll use every matfile in that dir and subdirs
      ncols      - number of columns to randomly mask
      win        - windows around columns to reconstruct
    """
    raise DeprecationWarning
    MINLENGTH = 50
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs = []
    # iterate
    for matfileidx, matfile in enumerate(matfiles):
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=30)
        # reconstruction
        W, Z, H, norm, recon, logprob = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                                                            rank,mask,win=win)
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        errs.append( err )
        total_cnt += 1
        # some info to pass time
        if matfileidx % 10000 == 0 and matfileidx > 0:
            print matfileidx,') average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs),'(',np.std(errs),')'
