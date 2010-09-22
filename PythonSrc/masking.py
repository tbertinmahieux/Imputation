"""
Set of function to apply a masking to some beatchroma matrix
(or probably any feature matrix)

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np
import scipy as sio



def random_col_mask(btchroma,ncols,win=3,seed=123):
    """
    Returns a binary mask where 'ncols' columns are zero,
    chosen at random (see param 'seed')
    INPUT
      btchroma    - features matrix (usually 12 x nbeats)
      ncols       - number of columns to mask
      win         - available window around cut columns
      seed        - seed (columns chosen at random)
    RETURN
      mask        - binary matrix, same shape as btchroma
      cols        - indices of masked columns, ascending order
    """
    assert btchroma.shape[1] > ncols,'too many columns too mask'
    np.random.seed(seed)
    if win % 2 == 0:
        win += 1
    half_win = int(np.floor(win/2))
    mask = np.ones(btchroma.shape)
    # choose columns
    masked_cols = np.array( range(half_win,btchroma.shape[1]-half_win) )
    np.random.shuffle(masked_cols)
    masked_cols = np.sort( masked_cols[:ncols] )
    # mask
    mask[:,masked_cols] = 0
    # done
    return mask, masked_cols


def random_patch_mask(btchroma,ncols,win=30,seed=123456):
    """
    Returns a binary mask where 'ncols' successive columns are zero,
    chosen at random (see param 'seed')
    INPUT
      btchroma    - features matrix (usually 12 x nbeats)
      ncols       - number of columns to mask
      win         - available beats around the masked patches
      seed        - seed (columns chosen at random)
    RETURN
      mask        - binary matrix, same shape as btchroma
      p1          - mask between cols p1 and p2
      p2            so mask[:,p1:p2] = 0, 1 everywhere else
    """
    assert btchroma.shape[1] > win * 2 + ncols,'btchroma too short'
    np.random.seed(seed)
    possible_p1 = range(win,btchroma.shape[1]-win-ncols)
    np.random.shuffle(possible_p1)
    p1 = possible_p1[0]
    p2 = p1 + ncols
    mask = np.ones(btchroma.shape)
    mask[:,p1:p2] = 0.
    return mask,p1,p2


def get_masked_cols(colmask):
    """
    Infer the indices of the columns of the mask
    INPUT
      colmask  - binary column mask, shape of a feature matrix
    RETURN
      cols     - indices of masked columns, ascending order
    """
    res = np.where(mask[0,:]==0)[0]
    assert (mask[:,res]==0).all(),'not a column mask'
    return res



def euclidean_dist_sq(v1,v2):
    """
    Trivial averaged squared euclidean distance from too flatten vectors
    """
    return np.square(v1.flatten()-v2.flatten()).mean()


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
        raise NotImplementedError
    else:
        raise ValueError('wrong measure name, want eucl or kl?')
    # measure and done
    maskzeros = np.where(mask==0)
    return measfun( btchroma[maskzeros] , recon[maskzeros] )



