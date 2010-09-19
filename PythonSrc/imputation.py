"""
Set of methods to do simple imputation, using correlation for
instance. Used as comparison to more complex methods like NMF.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""


import os
import sys
import numpy as np
import scipy.io as sio



def masked_eucl_dist_sq(f1,mask1,f2,mask2):
    """
    Computes covariance between two feature matrices,
    taking into accound the mask
    f1 and f2 same size. Not particularly fast.
    INPUT
      f1      - feature matrix
      mask1   - binary mask (same shape as f1)
      f2      - feature matrix (same shape as f1)
      mask2   - binary mask (same shape as f1)
    RETURN
      div     - average divergence measure
    """
    # sanity checks
    if f1.shape != f2.shape:
        print 'f1/f2 shapes:',f1.shape,f2.shape
    assert f1.shape == f2.shape,'bad feature matrix shape'
    assert f1.shape == mask1.shape,'bad mask 1 shape'
    assert f1.shape == mask2.shape,'bad mask 2 shape'
    # compute, and done
    maskones = np.where(mask1*mask2==1)
    return np.square(f1[maskones]-f2[maskones]).mean()


def average_col(btchroma,mask,masked_cols,win=21):
    """
    Fill in the mask based on the average column in the window
    around it.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      win          - window around column to consider for correlation (odd)
    RETURN
      recon        - full reconstruction, same size as btchroma
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert win > 1,'window of size 1 has no sense'
    if win % 2 == 0:
        win += 1
    # some inits
    full_recon = btchroma.copy()
    half_win = int(np.floor(win/2))
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        winstart = max(0, col - half_win)
        winstop = min(btchroma.shape[1], col + half_win + 1)
        patch = btchroma[:,winstart:winstop]
        patch_mask = mask[:,winstart:winstop]
        # compute averagerecon = np.zeros((btchroma.shape[0],1))
        full_recon[:,col] = patch[:,patch_mask[0,:]==1].mean(axis=1)
    # done
    return full_recon



def random_col(btchroma,mask,masked_cols):
    """
    Fill in the mask based on random
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
    RETURN
      recon        - full reconstruction, same size as btchroma
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    # some inits
    full_recon = btchroma.copy()
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        full_recon[:,col] = np.random.rand( btchroma.shape[0],1 ).flatten()
    # done
    return full_recon
        

def codebook_cols(btchroma,mask,masked_cols,codebook,measure='eucl'):
    """
    Fill in the missing columns using patches from a codebook.
    Patches are all the same size (12xWIN)
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      measure      - 'eucl' (euclidean distance, default)
                   - 'kl' (symmetric KL-divergence)
    RETURN
      recon        - full reconstruction, same size as btchroma
      used_codes   - code index used for reconstruction 
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    win = codebook[0].shape[1]
    # some inits
    if measure == 'eucl':
        measfun = masked_eucl_dist_sq
    elif measure == 'kl':
        raise NotImplementedError
    else:
        raise ValueError('wrong measure name, want eucl or kl?')
    full_recon = btchroma.copy()
    used_codes = np.zeros((1,len(masked_cols)))
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        # create subpatches with hidden column at each position
        # keep the 'full ones', avoid border effects
        patches = map(lambda x: btchroma[:,col-win+1+x:col+1+x],range(win))
        patches_masks = map(lambda x: mask[:,col-win+1+x:col+1+x],range(win))
        masked_col_index = range(win-1,0,-1)
        full_sized = filter(lambda i: patches[i].shape[1] == win, range(len(patches)))
        patches = map(lambda i: patches[i], full_sized)
        patches_masks = map(lambda i: patches_masks[i], full_sized)
        masked_col_index = filter(lambda i: masked_col_index, full_sized)
        assert len(patches) == len(patches_masks),'wrong creation of subpatches'
        # compare with every codeword
        best_div = np.inf
        best_code_idx = -1
        best_code_masked_col = -1
        for pidx,patch in enumerate(patches):
            divs = map(lambda code: measfun(patch,patches_masks[pidx],code,np.ones(code.shape)), codebook)
            divs = np.array(divs)
            codeidx = np.argmin(divs)
            if divs[codeidx] < best_div:
                best_div = divs[codeidx]
                best_code_idx = codeidx
                best_code_masked_col = masked_col_index[pidx]
        # done
        full_recon[:,col] = codebook[best_code_idx][:,best_code_masked_col].flatten()
        used_codes[colidx] = best_code_idx
    # done
    return full_recon, used_codes


def eucldist_cols(btchroma,mask,masked_cols,win=21):
    """
    Does imputation for columns by simple euclidean distance
    with the rest of the data.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      win          - window around column to consider for correlation (odd)
    RETURN
      recon        - full reconstruction, same size as btchroma
      used_cols    - columns used for reconstruction
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert win > 1,'window of size 1 has no sense'
    if win % 2 == 0:
        win += 1
    # some inits
    full_recon = btchroma.copy()
    half_win = int(np.floor(win/2))
    used_cols = np.zeros((1,len(masked_cols)))
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        winstart = max(0, col - half_win)
        winstop = min(btchroma.shape[1], col + half_win + 1)
        patch = btchroma[:,winstart:winstop]
        patch_mask = mask[:,winstart:winstop]
        # best correlation
        best_div = np.inf
        nunmask = -np.inf   # if same divergence, prefer the one that matches more pixels
        recon = np.zeros((btchroma.shape[0],1))
        # perform 'same' correlation
        for c in range(btchroma.shape[1]):
            if len(np.where(masked_cols==c)[0]) > 0:
                continue # useless if we match a masked column
            reconwinstart =  max(0, c - half_win)
            reconwinstop = min(btchroma.shape[1], c+half_win+1)
            reconpatch = btchroma[:,reconwinstart:reconwinstop]
            reconpatch_mask = mask[:,reconwinstart:reconwinstop]
            # if we use onlypart of the patch, for border issues
            p1 = half_win-c+reconwinstart # 0 if full patch used
            p2 = p1 + reconpatch.shape[1] # patch.shape[1] if full patch used
            div = masked_eucl_dist_sq(patch[:,p1:p2],patch_mask[:,p1:p2],
                                      reconpatch,reconpatch_mask)
            # check if got better divergence
            if div < best_div:
                best_div = div
                recon = btchroma[:,c]
                nunmask = len(np.where(reconpatch_mask==1)[0])
                used_cols[0,colidx] = c
            elif div == best_div and len(np.where(reconpatch_mask==1)[0]) > nunmask:
                recon = btchroma[:,c]
                nunmask = len(np.where(reconpatch_mask==1)[0])
                used_cols[0,colidx] = c
        # iteration done
        assert not np.isinf(best_div),'iteration did not work? best_corr is inf'
        assert not np.isnan(best_div),'computation went wrong, best_corr is NaN'
        assert not np.isinf(nunmask),'computation went wrong, best_corr is NaN'
        full_recon[:,col] = recon
    # done
    return full_recon, used_cols


