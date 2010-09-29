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
import scipy.spatial.distance as DIST

import imputation as IMPUTATION
import imputation_plca as IMPUTATION_PLCA
import hmm_imputation as IMPUTATION_HMM
try:
    import kalman_toolbox as KALMAN
except ImportError:
    print 'cant import KALMAN: no matlab maybe?'
import masking as MASKING

EPS = np.finfo(np.float).eps


def euclidean_dist_sq(v1,v2):
    """
    Trivial averaged squared euclidean distance from too flatten vectors
    """
    return np.square(v1.flatten()-v2.flatten()).mean()

def cosine_dist(v1,v2):
    """
    Consine distance between too flatten vector
    """
    return DIST.cosine(v1.flatten(),v2.flatten())

def symm_kl_div(v1,v2):
    """
    Normalize and return the symmetric kldivergence
    """
    v1 = v1.copy() / v1.sum() + EPS
    v2 = v2.copy() / v2.sum() + EPS
    assert not np.isnan(v1).any()
    assert not np.isnan(v2).any()
    res1 = (v1 * np.log(v1 / v2))
    res2 = (v2 * np.log(v2 / v1))
    res1[np.where(np.isnan(res1))] = 0.
    res2[np.where(np.isnan(res2))] = 0.
    div1 = res1.sum()
    div2 = res2.sum()
    return (div1 + div2) / 2.


def recon_error(btchroma,mask,recon,measure='eucl'):
    """

    INPUT
       btchroma   - original feature matrix
       mask       - binary mask, same shape as btchroma
       recon      - reconstruction, same shape as btchroma
       measure    - 'eucl' (euclidean distance, default)
                  - 'kl' (symmetric KL-divergence)
                  - 'cos' (cosine distance)
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
    elif measure == 'cos':
        measfun = cosine_dist
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



def test_maskedcol_on_dataset(datasetdir,method='random',ncols=1,win=3,rank=4,codebook=None,**kwargs):
    """
    General method to test a method on a whole dataset for one masked column
    Methods are:
      - random
      - randomfromsong
      - average
      - codebook
      - knn_eucl
      - knn_kl
      - lintrans
      - siplca
      - siplca2
    Used arguments vary based on the method. For SIPLCA, we can use **kwargs
    to set priors.
    """
    MINLENGTH = 70
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs_eucl = []
    errs_kl = []
    # some specific inits
    if codebook != None and not type(codebook) == type([]):
        codebook = [p.reshape(12,codebook.shape[1]/12) for p in codebook]
        print 'codebook in ndarray format transformed to list'
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,masked_cols = MASKING.random_col_mask(btchroma,ncols=ncols,win=25)
        ########## ALGORITHM DEPENDENT
        if method == 'random':
            recon = IMPUTATION.random_col(btchroma,mask,masked_cols)
        elif method == 'randomfromsong':
            recon = IMPUTATION.random_col_from_song(btchroma,mask,masked_cols)
        elif method == 'average':
            recon = IMPUTATION.average_col(btchroma,mask,masked_cols,win=win)
        elif method == 'codebook':
            recon,used_codes = IMPUTATION.codebook_cols(btchroma,mask,masked_cols,codebook)
        elif method == 'knn_eucl':
            recon,used_cols = IMPUTATION.knn_cols(btchroma,mask,masked_cols,win=win,measure='eucl')
        elif method == 'knn_kl':
            recon,used_cols = IMPUTATION.knn_cols(btchroma,mask,masked_cols,win=win,measure='kl')
        elif method == 'lintrans':
            recon,proj = IMPUTATION.lintransform_cols(btchroma,mask,masked_cols,win=win)
        elif method == 'siplca':
            res = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                                      rank,mask,win=win,
                                                      convergence_thresh=1e-15,
                                                      **kwargs)
            W, Z, H, norm, recon, logprob = res
        elif method == 'siplca2':
            res = IMPUTATION_PLCA.SIPLCA2_mask.analyze((btchroma*mask).copy(),
                                                       rank,mask,win=win,
                                                       convergence_thresh=1e-15,
                                                       **kwargs)
            W, Z, H, norm, recon, logprob = res
        else:
            print 'unknown method:',method
            return
        ########## ALGORITHM DEPENDENT END
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        if err > 100:
            print 'huge EUCL error:',err,', method =',method,',file =',matfile
        errs_eucl.append( err )
        err = recon_error(btchroma,mask,recon,measure='kl')
        if err > 100:
            print 'huge KL error:',err,', method =',method,',file =',matfile
        errs_kl.append( err )
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs_eucl),'(',np.std(errs_eucl),')'
    print 'average kl divergence:',np.mean(errs_kl),'(',np.std(errs_kl),')'


def test_maskedpatch_on_dataset(datasetdir,method='random',ncols=2,win=1,rank=4,codebook=None,nstates=1,**kwargs):
    """
    General method to test a method on a whole dataset for one masked column
    Methods are:
      - random
      - randomfromsong
      - average
      - averageall
      - codebook
      - knn_eucl
      - knn_kl
      - lintrans
      - kalman
      - hmm
      - siplca
      - siplca2
    Used arguments vary based on the method. For SIPLCA, we can use **kwargs
    to set priors.
    """
    MINLENGTH = 70
    # get all matfiles
    matfiles = get_all_matfiles(datasetdir)
    # init
    total_cnt = 0
    errs_eucl = []
    errs_kl = []
    # some specific inits
    if codebook != None and not type(codebook) == type([]):
        codebook = [p.reshape(12,codebook.shape[1]/12) for p in codebook]
        print 'codebook in ndarray format transformed to list'
    # iterate
    for matfile in matfiles:
        btchroma = sio.loadmat(matfile)['btchroma']
        if btchroma.shape[1] < MINLENGTH or np.isnan(btchroma).any():
            continue
        mask,p1,p2 = MASKING.random_patch_mask(btchroma,ncols=ncols,win=25)
        ########## ALGORITHM DEPENDENT
        if method == 'random':
            recon = IMPUTATION.random_patch(btchroma,mask,p1,p2)
        elif method == 'randomfromsong':
            recon = IMPUTATION.random_patch_from_song(btchroma,mask,p1,p2)
        elif method == 'average':
            recon = IMPUTATION.average_patch(btchroma,mask,p1,p2,win=win)
        elif method == 'averageall':
            recon = IMPUTATION.average_patch_all(btchroma,mask,p1,p2)
        elif method == 'codebook':
            recon,used_codes = IMPUTATION.codebook_patch(btchroma,mask,p1,p2,codebook)
        elif method == 'knn_eucl':
            recon,used_cols = IMPUTATION.knn_patch(btchroma,mask,p1,p2,win=win,measure='eucl')
        elif method == 'knn_kl':
            recon,used_cols = IMPUTATION.knn_patch(btchroma,mask,p1,p2,win=win,measure='kl')
        elif method == 'lintrans':
            recon,proj = IMPUTATION.lintransform_patch(btchroma,mask,p1,p2,win=win)
        elif method == 'kalman':
            recon = KALMAN.imputation(btchroma,p1,p2,
                                      dimstates=nstates,**kwargs)
        elif method == 'hmm':
            recon,recon2,hmm = IMPUTATION_HMM.imputation(btchroma*mask,p1,p2,
                                                         nstates=nstates,
                                                         **kwargs)
        elif method == 'siplca':
            res = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                                      rank,mask,win=win,
                                                      convergence_thresh=1e-15,
                                                      **kwargs)
            W, Z, H, norm, recon, logprob = res
        elif method == 'siplca2':
            res = IMPUTATION_PLCA.SIPLCA2_mask.analyze((btchroma*mask).copy(),
                                                       rank,mask,win=win,
                                                       convergence_thresh=1e-15,
                                                       **kwargs)
            W, Z, H, norm, recon, logprob = res
        else:
            print 'unknown method:',method
            return
        ########## ALGORITHM DEPENDENT END
        # measure recon
        err = recon_error(btchroma,mask,recon,measure='eucl')
        if err > 100:
            print 'huge EUCL error:',err,', method =',method,',file =',matfile
        errs_eucl.append( err )
        err = recon_error(btchroma,mask,recon,measure='kl')
        errs_kl.append( err )
        if err > 100:
            print 'huge KL error:',err,', method =',method,',file =',matfile
        total_cnt += 1
    # done
    print 'number of songs tested:',total_cnt
    print 'average sq euclidean dist:',np.mean(errs_eucl),'(',np.std(errs_eucl),')'
    print 'average kl divergence:',np.mean(errs_kl),'(',np.std(errs_kl),')'



    
def cut_train_test_by_numbers(datasetdir,nums=[5000,5000],seed=666999):
    """
    Get all songs from datasetdir.
    Order them by name, then shuffle (we know the seed!)
    Return two list of files, for train and for test, according to numbers
    in nums.
    Of course, files are not overlapping.
    If we change one number, the same files remained used for the other set
    (after shuffling, train is taken from the beggining of the list,
    test is taken from the end).
    """
    # sanity check
    assert len(nums)==2,'we need a list of two numbers'
    # get all matfiles
    matfiles = map(lambda x: os.path.abspath(x),get_all_matfiles(datasetdir))
    # check if we have enough files
    assert len(matfiles)>=nums[0] + nums[1],'not enough files for number requested'
    # order by name
    matfiles = np.sort(matfiles)
    # shuffle
    np.random.seed(seed)
    np.random.shuffle(matfiles)
    # cut
    train = list( matfiles[:nums[0]] )
    test = list( matfiles[-nums[1]:] )
    # done
    return train, test

