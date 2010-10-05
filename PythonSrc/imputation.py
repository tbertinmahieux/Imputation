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
import linear_transform as LINTRANS
import evaluation as EVAL

EPS = np.finfo(np.float).eps


def masked_eucl_dist_sq(f1,mask1,f2,mask2):
    """
    Computes euclideand distance between two feature matrices,
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
    if len(maskones[0]) == 0:
        return np.nan # combination of masks mask everything'
    res = np.square(f1[maskones]-f2[maskones]).mean()
    assert not np.isnan(res),'euclidean distance computes NaN' 
    return res


def masked_kl_div(f1,mask1,f2,mask2):
    """
    Computes symmetric KL divergence between two feature matrices,
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
    if len(maskones[0]) == 0:
        return np.nan # combination of masks mask everything'
    return EVAL.symm_kl_div(f1[maskones],f2[maskones])


def average_col(btchroma,mask,masked_cols,win=3):
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

def average_patch(btchroma,mask,p1,p2,win=1):
    """
    Infer the values of a patch as a constant function, made of the
    average of the neighborhood beats.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      p1           - beginning of the masked patch
      p2           - end of the mask patch, mask[:,p1:p2] = 0
      win          - window to consider on eahc side of the patch
    RETURN
      recon        - full reconstruction, same size as btchroma
    If the patch is one column, average_patch with win = 1 works
    the same way as average_col with win = 3
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert win >= 1,'window of size 0 has no sense'
    # some inits
    full_recon = btchroma.copy()
    # neighborhood beats
    neigh_beats = np.concatenate( [btchroma[:,p1-win:p1],
                                   btchroma[:,p2:p2+win]], axis=1)
    meanbeat = neigh_beats.mean(axis=1).flatten()
    for k in range(p1,p2):
        full_recon[:,k] = meanbeat
    # done
    return full_recon

def average_patch_all(btchroma,mask,p1,p2,win=1):
    """
    Infer the values of a patch as a constant function, made of the
    average of all the visible beats of the song.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      p1           - beginning of the masked patch
      p2           - end of the mask patch, mask[:,p1:p2] = 0
      win          - window to consider on eahc side of the patch
    RETURN
      recon        - full reconstruction, same size as btchroma
    If the patch is one column, average_patch with win = 1 works
    the same way as average_col with win = 3
    """
     # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert win >= 1,'window of size 0 has no sense'
    # some inits
    full_recon = btchroma.copy()
    avg_beat = np.concatenate([btchroma[:,:p1],btchroma[:,p2:]],
                              axis=1).mean(axis=1).flatten()
    # iter over missing columns
    for k in range(p1,p2):
        full_recon[:,k] = avg_beat
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

def random_patch(btchroma,mask,p1,p2):
    """ see random col """
    return random_col(btchroma,mask,range(p1,p2))


def random_col_from_song(btchroma,mask,masked_cols):
    """
    Fill in the mask by copying a random column from the song
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
    # all columns in a set
    allcols = set(range(btchroma.shape[1]))
    # remove masked cols
    map(lambda x: allcols.remove(x), masked_cols)
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        rand_col = list(allcols)[np.random.randint(len(allcols))]
        full_recon[:,col] = btchroma[:,rand_col].flatten()
    # done
    return full_recon

def random_patch_from_song(btchroma,mask,p1,p2):
    """
    Fill in the mask by copying a random patch from the song
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
    RETURN
      recon        - full reconstruction, same size as btchroma
    if patch ncols = 1, same as random_col_form_song
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    # some inits
    full_recon = btchroma.copy()
    # check such patch exists
    ncols = p2-p1
    leftsidelen = p1
    rightsidelen = btchroma.shape[1] - p2
    assert leftsidelen >= ncols or rightsidelen >= ncols,'cant find large enough patch'
    # choose one at random
    possible_patches_start = range(0,leftsidelen-ncols)
    possible_patches_start.extend( range(p2,btchroma.shape[1]-ncols) )
    np.random.shuffle(possible_patches_start)
    randstart = possible_patches_start[0]
    full_recon[:,p1:p2] = btchroma[:,randstart:randstart+ncols].copy()
    # done
    return full_recon


def codebook_cols(btchroma,mask,masked_cols,codebook,measure='eucl',userecon=False):
    """
    Fill in the missing columns using patches from a codebook.
    Patches are all the same size (12xWIN)
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      measure      - 'eucl' (euclidean distance, default)
                   - 'kl' (symmetric KL-divergence)
                   - 'cos' (cosine distance)
      userecon     - if True, as soon as a col is added, it can be used
                     necessary if codes smaller than gap
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
        measfun = masked_kl_div
    else:
        raise ValueError('wrong measure name, want eucl or kl?')
    full_recon = btchroma.copy()
    used_codes = np.zeros((1,len(masked_cols)))
    # iterate over masked columns
    for colidx,col in enumerate(np.array(masked_cols).copy()): # copy important if userecon
        # create subpatches with hidden column at each position
        # keep the 'full ones', avoid border effects
        patches = map(lambda x: btchroma[:,col-win+1+x:col+1+x],range(win))
        patches_masks = map(lambda x: mask[:,col-win+1+x:col+1+x],range(win))
        # where is the hidden col in those patches
        masked_col_index = range(win-1,-1,-1)
        assert len(masked_col_index) == len(patches)
        # keep the full ones, no border effect
        full_sized = filter(lambda i: patches[i].shape[1] == win, range(len(patches))) # indeces
        patches = map(lambda i: patches[i], full_sized)
        patches_masks = map(lambda i: patches_masks[i], full_sized)
        masked_col_index = map(lambda i: masked_col_index[i], full_sized)
        assert len(patches) == len(patches_masks),'wrong creation of subpatches'
        assert len(patches) == len(masked_col_index),'wrong creation of subpatches'
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
        assert best_code_idx >= 0,'no proper code found/use, window prob?'
        # done
        full_recon[:,col] = codebook[best_code_idx][:,best_code_masked_col].flatten()
        used_codes[0,colidx] = best_code_idx
        # use recon?
        if userecon:
            btchroma = full_recon.copy()
            mask = mask.copy()
            mask[:,col] = 1
    # done
    return full_recon, used_codes


def codebook_patch(btchroma,mask,p1,p2,codebook=None):
    """ see codebook_cols """
    assert codebook is not None,'we need a codebook!'
    return codebook_cols(btchroma,mask,range(p1,p2),codebook,userecon=True)



def knn_cols(btchroma,mask,masked_cols,win=21,measure='eucl'):
    """
    Does imputation for columns by simple euclidean distance
    with the rest of the data.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      win          - window around column to consider for correlation (odd)
      measure      - 'eucl' or 'kl', for euclidean distance or KL-divergence
    RETURN
      recon        - full reconstruction, same size as btchroma
      used_cols    - columns used for reconstruction
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert not np.isnan(btchroma).any(),'btchroma has NaN'
    assert win > 1,'window of size 1 has no sense'
    if win % 2 == 0:
        win += 1
    # some inits
    if measure == 'eucl':
        distfun=masked_eucl_dist_sq
    elif measure == 'kl':
        distfun=masked_kl_div
    else:
        print 'wrong measure:',measure
        return None
    full_recon = btchroma.copy()
    half_win = int(np.floor(win/2))
    used_cols = np.zeros((1,len(masked_cols)))
    # iterate over masked columns
    for colidx,col in enumerate(masked_cols):
        winstart = max(0, col - half_win)
        winstop = min(btchroma.shape[1], col + half_win + 1)
        patch = btchroma[:,winstart:winstop]
        patch_mask = mask[:,winstart:winstop]
        if not (patch_mask==1).any():
            print 'column impossible to fill: col =',col,' btchroma.shape =',btchroma.shape,' masked_cols =',masked_cols,' win=',win
        assert (patch_mask==1).any(),'full mask! useless'
        # best correlation
        best_div = np.inf
        nunmask = -np.inf   # if same divergence, prefer the one that matches more pixels
        recon = np.zeros((btchroma.shape[0],1))
        # perform knn
        for c in range(btchroma.shape[1]):
            if len(np.where(np.array(masked_cols)==c)[0]) > 0:
                continue # useless if we match a masked column
            reconwinstart =  max(0, c - half_win)
            reconwinstop = min(btchroma.shape[1], c+half_win+1)
            reconpatch = btchroma[:,reconwinstart:reconwinstop]
            reconpatch_mask = mask[:,reconwinstart:reconwinstop]
            # if we use only part of the patch, for border issues
            p1 = half_win-c+reconwinstart # 0 if full patch used
            p2 = p1 + reconpatch.shape[1] # patch.shape[1] if full patch used
            if not (patch_mask[:,p1:p2]==1).any(): #'patch_mask subset all masked, useless'
                continue
            div = distfun(patch[:,p1:p2],patch_mask[:,p1:p2],
                          reconpatch,reconpatch_mask)
            if np.isnan(div):
                continue
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
        assert not np.isinf(best_div),'iteration did not work? best_div is inf'
        assert not np.isnan(best_div),'computation went wrong, best_div is NaN'
        assert not np.isinf(nunmask),'computation went wrong, nunmask is NaN'
        full_recon[:,col] = recon
    # done
    return full_recon, used_cols

def knn_patch(btchroma,mask,p1,p2,win=21,measure='eucl'):
    """ see knn_col, win is now window on each side """
    return knn_cols(btchroma,mask,range(p1,p2),win=win*2+1,measure=measure)

def knn_patch_delta(btchroma,mask,p1,p2,win=21,measure='eucl'):
    """
    see knn_col, win is now window on each side
    Works on delta features!!! rowwise diff of the original
    """
    V = (btchroma*mask).copy()
    Vdiff = np.concatenate([np.zeros((12,1)),np.diff(V)],axis=1)
    assert Vdiff.shape == V.shape,'bad creation of btchroma delta'
    # do NN
    recon,used_cols = knn_cols(V,mask,range(p1,p2),win=win*2+1,measure=measure)
    # build result
    for col in range(p1,p2):
        V[:,col] = V[:,col-1] + recon[:,col]
    V[np.where(V>1.)] = 1.
    V[np.where(V<0.)] = 0.
    assert V.max() <= 1.
    assert V.min() >= 0.
    return V, used_cols
    

def lintransform_cols(btchroma,mask,masked_cols,win=1):#,niter=1000,lrate=.1):
    """
    Does imputation for columns by simple euclidean distance
    with the rest of the data.
    INPUT
      btchroma     - feature matrix
      mask         - binary mask
      masked_cols  - indices of the masked columns
      win          - window to consider before masked column
      niter        - number of iterations for training
      lrate        - learning rate for training
    RETURN
      recon        - full reconstruction, same size as btchroma
      lintrans     - learned linear transform class
    """
    # sanity checks
    assert btchroma.shape == mask.shape,'wrong mask shape'
    assert win > 0,'window of size 0,-1,... has no sense'
    # some init
    full_recon = btchroma.copy()
    # cut btchroma in every possible patches of size win+1
    btchromamasked = btchroma.copy()
    btchromamasked[np.where(mask==0)] = np.nan
    allpatches = map(lambda x: btchromamasked[:,x-win:x+1].copy(), range(win,btchroma.shape[1]))
    # remove patches containing masked columns
    prev_n_patches = len(allpatches)
    allpatches = filter(lambda x: not np.isnan(x).any(), allpatches)
    # cut into indata and outdata
    indata = map(lambda x: x[:,:win].flatten(), allpatches)
    outdata = map(lambda x: x[:,win:].flatten(), allpatches)
    # use linear regression
    indata = np.concatenate( [np.array(indata) , np.ones((len(indata),1))], axis=1)
    outdata = np.array(outdata)
    # learn linear projection
    proj = LINTRANS.solve_linear_equation(indata,outdata)
    for colidx, col in enumerate(masked_cols):
        #indata = np.concatenate([btchroma[:,col-win:col].flatten(),[1]])
        indata = np.concatenate([full_recon[:,col-win:col].flatten(),[1]])
        recon = np.dot(indata.reshape(1,indata.size) , proj)
        recon[np.where(recon>1.)] = 1.
        recon[np.where(recon<EPS)] = EPS
        full_recon[:,col] = recon
    # done
    return full_recon, proj

def lintransform_patch(btchroma,mask,p1,p2,win=1):
    """ see lintransform_cols """
    return lintransform_cols(btchroma,mask,range(p1,p2),win=win)
