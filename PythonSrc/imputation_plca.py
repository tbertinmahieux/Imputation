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
try:
    import plca as PLCA
except ImportError:
    from ronwsiplca_cover import plca as PLCA
shift = PLCA.shift
logger = PLCA.logger


class SIPLCA_mask(PLCA.SIPLCA):
    """
    Sparse shift-invariant PLCA, with the option to have a
    binary mask during analysis
    """
    @classmethod
    def analyze(cls, V, rank, mask, niter=300, convergence_thresh=1e-9,
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
        recon = V.copy()
        maskzeros = np.where(mask==0)
        maskones = np.where(mask==1)
        norm_justones = V[maskones].sum()
        p1 = np.where(mask[0,:]==0)[0][0]
        p2 = np.where(mask[0,:]==0)[0][-1] + 1
        # approximate sum of missing data
        V = V.copy()
        norm = norm_justones * V.size / len(maskones[0])
        V /= norm
        # init with noise
        #V[maskzeros] = np.random.rand(len(maskzeros[0])) / V[maskones].max()
        V[maskzeros] = 0.
        # *****************
    
        params = cls(V, rank, **kwargs)
        iW, iZ, iH = params.initialize()
    
        W = iW if initW is None else initW.copy()
        Z = iZ if initZ is None else initZ.copy()
        H = iH if initH is None else initH.copy()
    
        params.W = W
        params.Z = Z
        params.H = H

        # second layer
        doH = True
        rankH = 2
        winH = 10
        modelH = PLCA.SIPLCA.analyze(H,rankH,niter=1,win=winH)
        #************************

        oldlogprob = -np.inf
        for n in xrange(niter):
            # ADDED FOR MASKING
            #WZH = cls.reconstruct_mask(W, Z, H, p1, p2)
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

            # second level on H!
            if doH:
                modelH = PLCA.SIPLCA.analyze(H,rankH,initW=modelH[0],
                                             initZ=modelH[1],initH=modelH[2],
                                             niter=1,win=winH)
                H = modelH[4]
            # **********************
            
            params.W = W
            params.Z = Z
            params.H = H

        if plotiter:
            params.plot(V, W, Z, H, n) 
            if not plotfilename is None:
                plt.savefig('%s_%04d.png' % (plotfilename, n))
        if niter > 0:
            logger.info('Iteration %d: final logprob = %f', n, logprob)
        # ADDED FOR MASKING
        # norm can be slightly wrong from missing data
        #norm = norm_justones / params.V[maskones].sum() # empirically not good
        # *****************
        reconall = norm * WZH
        recon[maskzeros] = reconall[maskzeros]
        # fix zeroz and ones
        recon[np.where(recon>1.)] = 1.
        return W, Z, H, norm, recon, logprob

    @staticmethod
    def reconstruct_mask(W, Z, H, p1,p2, norm=1.0, circular=False):
        if W.ndim == 2:
            W = W[:,np.newaxis,:]
        if H.ndim == 1:
            H = H[np.newaxis,:]
        F, rank, win = W.shape
        rank, T = H.shape
        # lets work on H
        #H = H.copy() # shape 4x218 if rank=4,length=218
        max_per_rank = np.concatenate([H[:,:p1],H[:,p2:]],axis=1).max(1)
        for p in xrange(p1,p2):
            for r in xrange(rank):
                H[r,p] = min(H[r,p],max_per_rank[r])
        #***************
        WZHs = np.zeros((F, T, win))
        for tau in xrange(win):
            WZHs[:,:,tau] = np.dot(W[:,:,tau] * Z, shift(H, tau, 1, circular))
        # take the sum, except missing part where we take the max
        WZH = WZHs.sum(axis=2) # or not
        return norm * WZH


class SIPLCA2_mask(PLCA.SIPLCA2):
    """
    Sparse shift-invariant PLCA (in 2 directions),
    with the option to have a binary mask during analysis
    Should be copy/paste of code from SIPLCA_mask
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
        recon = V.copy()
        maskzeros = np.where(mask==0)
        maskones = np.where(mask==1)
        norm_justones = V[maskones].sum()
        # approximate sum of missing data
        V = V.copy()
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
        reconall = norm * WZH
        recon[maskzeros] = reconall[maskzeros]
        return W, Z, H, norm, recon, logprob
