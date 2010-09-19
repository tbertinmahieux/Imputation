"""
Classes and function that derive from Ron's PLCA implementation,
but adds the option to maks data.
We can imputation from there.
Youppie :)

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np

# Ron's SIPLCA
#sys.path.append( os.path.expanduser('~/Columbia/ronwsiplca_cover') )
import plca as PLCA
logger = PLCA.logger


class SIPLCA_mask(PLCA.SIPLCA):
    """
    Sparse shift-invariant PLCA, with the option to have a
    binary mask during analysis
    """

    @classmethod
    def analyze(cls, V, rank, mask, niter=100, convergence_thresh=1e-9,
                printiter=50, plotiter=None, plotfilename=None,
                initW=None, initZ=None, initH=None,
                updateW=True, updateZ=True, updateH=True,**kwargs):
        """Iteratively performs the PLCA decomposition using the EM algorithm

        Parameters
        ----------
        V : array, shape (`F`, `T`)
            Matrix to analyze.
        mask : array, shape (`F`, `T`)
            Binarymask, 0 for missing params in V
        niter : int
            Number of iterations to perform.  Defaults to 100.
        convergence_thresh : float
        updateW, updateZ, updateH : boolean
            If False keeps the corresponding parameter fixed.
            Defaults to True.
        initW, initZ, initH : array
            Initial settings for `W`, `Z`, and `H`.  Unused by default.
        printiter : int
            Prints current log probability once every `printiter`
            iterations.  Defaults to 50.
        plotiter : int or None
            If not None, the current decomposition is plotted once
            every `plotiter` iterations.  Defaults to None.
        kwargs : dict
            Arguments to pass into the class's constructor.

        Returns
        -------
        W : array, shape (`F`, `rank`)
            Set of `rank` bases found in `V`, i.e. P(f | z).
        Z : array, shape (`rank`)
            Mixing weights over basis vector activations, i.e. P(z).
        H : array, shape (`rank`, `T`)
            Activations of each basis in time, i.e. P(t | z).
        norm : float
            Normalization constant to make `V` sum to 1.
        recon : array
            Reconstruction of `V` using `W`, `Z`, and `H`
        logprob : float
        """
        #norm = V.sum()
        #V /= norm

        # ADDED FOR MASKING
        maskzeros = np.where(mask==0)
        maskones = np.where(mask==1)
        norm_justones = V[maskones].sum()
        # approximate sum of missing data
        norm = norm_justones * V.size / len(maskones[0])
        V /= norm
        # *****************
    
        params = cls(V, rank, **kwargs)
        iW, iZ, iH = params.initialize()
    
        W = iW if initW is None else initW.copy()
        Z = iZ if initZ is None else initZ.copy()
        H = iH if initH is None else initH.copy()
    
        params.W = W
        params.Z = Z
        params.H = H

        oldlogprob = -np.inf
        for n in xrange(niter):
            # ADDED FOR MASKING
            WZH = cls.reconstruct(W, Z, H)
            V[maskzeros] = WZH[maskzeros]
            params.V = V
            # *****************
            logprob, WZH = params.do_estep(W, Z, H)
            if n % printiter == 0:
                logger.info('Iteration %d: logprob = %f', n, logprob)
            if plotiter and n % plotiter == 0:
                params.plot(V, W, Z, H, n)
                if not plotfilename is None:
                    plt.savefig('%s_%04d.png' % (plotfilename, n))
            if logprob < oldlogprob:
                logger.debug('Warning: logprob decreased from %f to %f at '
                             'iteration %d!', oldlogprob, logprob, n)
                #import pdb; pdb.set_trace()
            elif n > 0 and logprob - oldlogprob < convergence_thresh:
                logger.info('Converged at iteration %d', n)
                break
            oldlogprob = logprob
    
            nW, nZ, nH = params.do_mstep(n)
    
            if updateW:  W = nW
            if updateZ:  Z = nZ
            if updateH:  H = nH
    
            params.W = W
            params.Z = Z
            params.H = H

        if plotiter:
            params.plot(V, W, Z, H, n)
            if not plotfilename is None:
                plt.savefig('%s_%04d.png' % (plotfilename, n))
        logger.info('Iteration %d: final logprob = %f', n, logprob)
        # ADDED FOR MASKING
        # norm can be slightly wrong from missing data
        norm = norm_justones / params.V[maskones].sum()
        # *****************
        recon = norm * WZH
        return W, Z, H, norm, recon, logprob


