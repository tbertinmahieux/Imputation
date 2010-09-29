"""
Set of functions to do clustering, usually kmeans, on a bunch
of btcroma.

It reproduces our ISMIR 2010 paper, wihtout the online thing.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""


import os
import sys
import numpy as np
import scipy.io as sio

import scipy.cluster.vq as VQ



def do_kmeans(data,nclusters=2,niters=2,centroids=None):
    """
    Wrapper around scipy.cluster.vq.kmeans2

    INPUT
      data       - one example per row
      nclusters  - k of kmeans
      niters     - how long kmeans will run
      centroids  - result of a previous run, to continue training

    RETURN
      centroids  - the 'nclusters' means, as one array, order by frequency
                   on training data (descending order)
      labels     - labels[i] is the index of the centroid the
                   i'th observation is closest to.
                   
    NOTE: kmeans  - iter is the number of times to run kmeans,
                    stop because of threshold
          kmeans2 - iter is the number of iteration, no threshold
                    (even if param 'thresh' exists, it is not used!)
    """
    if centroids == None:
        cb,labels =  VQ.kmeans2(data,nclusters,iter=niters,
                                minit='random',missing='warn')
    else:
        assert nclusters == centroids.shape[0],'adjust nclusters so it fits centroids'
        cb,labels =  VQ.kmeans2(data,centroids,iter=niters,
                                minit='matrix',missing='warn')
    # reorder so most frequent code comes first
    return order_codebook_by_freq(cb,labels)


def crop_and_split(btchroma,nbeats,flatten=True):
    """
    Helper function, cut a btchroma into slices of nbeats,
    non overlapping, crop the end if needed
    Return as a list, result of numpy.split
    If flatten, flatten each piece to a 1xNCHROMAS*NBEATS vector
    """
    ncuts = int(btchroma.shape[1] / nbeats)
    slices = np.split(btchroma[:,:nbeats*ncuts],ncuts,axis=1)
    if flatten:
        nc = btchroma.shape[0]
        return map(lambda s: s.reshape(1,nc*nbeats),slices)
    else:
        return slices

def create_cluster_data(matfiles,nbeats,overlap=False):
    """
    Get the beatchroma from all files, create one big matrix
    with each slice of size nbeats, flatten it to one example
    per row

    INPUT
      matfiles  - list of files containing 'btchroma'
      nbeats    - number of beats per pattern (patch)
      overlap   - full overlap (every beat) or not at all

    RETURN
      data      - one patch per row, 12 x nbeats
    """
    # get all bthcroma
    btchromas = map(lambda x: sio.loadmat(x)['btchroma'],matfiles)
    # make sure each song has enough beats
    btchromas = filter(lambda btc: btc.shape[1] >= nbeats, btchromas)
    # get all slices
    if overlap:
        nc = btchromas[0].shape[0]
        slices = map(lambda btc:
                     np.concatenate( map(lambda x:
                                         btc[:,x:x+nbeats].reshape(1,nc*nbeats),
                                         range(btc.shape[1]-nbeats)),
                                     axis=0 ) , btchromas)
    else:
        slices = map(lambda btc:
                     np.concatenate( crop_and_split(btc,nbeats,True), axis=0 ),
                     btchromas)
    # create one big matrix
    return np.concatenate(slices,axis=0)


def order_codebook_by_freq(codebook,indeces):
    """
    Receives the output of do_means, and reorder the result
    so the most used codebook is at the beginning.

    INPUT
       codebook   - array, NxM, N number of codes, M=12xNBEATS
       indeces    - if K examples were seen, list of indeces
                    telling for each example which code was
                    the closest
    RETURN
       codebook   - codebook reordered
       indeces    - indeces updated for the new codebook order
    """
    # sanity check
    assert max(indeces) <= codebook.shape[0],'index inconsistency'
    # freqs
    freqs = np.zeros(codebook.shape[0])
    # sort, descending order
    order = np.flipud( np.argsort(freqs) )
    # reverse indeces
    old_i_to_new_i = np.zeros(codebook.shape[0])
    for i in range(codebook.shape[0]):
        old_i_to_new_i[i] = np.where(order==i)[0][0]
    # reorder codebook and indeces
    codebook = codebook[order,:]
    newindeces = np.zeros(len(indeces))
    for idx,i in enumerate(indeces):
        newindeces[idx] = old_i_to_new_i[i]
    # done
    return codebook, list(newindeces)



