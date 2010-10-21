"""
Does imputation of missing beats of chroma features using an HMM.
We use Ron's HMM implementation from scikits.learn

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import string
import numpy as np
import scipy.io as sio
from scikits.learn import hmm as HMM
from scikits.learn.gmm import (GMM,logsum)

ZEROLOGPROB = HMM.ZEROLOGPROB

#class GMMHMM_impute(HMM.GMMHMM):
class GMMHMM_impute(HMM.GMMHMM):
    """
    Same class as GMMHMM except for a slight modification
    of the M step.
    We also update the data between position p1 and p2
    to maximize likelihood. This is how we do imputation.

    n_dim is modified to 12 by default, for chroma features
    All the rest should remain the same
    """
    def __init__(self, impute_p1, impute_p2, n_states=1, n_dim=12,
                 n_mix=1, startprob=None,
                 transmat=None, startprob_prior=None, transmat_prior=None,
                 labels=None, gmms=None, cvtype=None):
        # save p1 and p2
        assert impute_p1 < impute_p2,'p1 and p2 in wrong order?'
        self.impute_p1 = impute_p1
        self.impute_p2 = impute_p2
        # class parent constructor
        super(GMMHMM_impute,self).__init__(n_states,n_dim,n_mix,startprob,
                                           transmat,startprob_prior,
                                           transmat_prior,labels,gmms,
                                           cvtype)

    def fit(self, obs, n_iter=10, thresh=1e-2, params=string.letters,
            init_params=string.letters,
            maxrank=None, beamlogprob=-np.Inf, **kwargs):
        """
        Slight modification of the fit function, just to give us
        a reference to the observation, so we can modify them
        (impute them) in the M step.
        """
        # keep observations
        self.curr_obs = obs
        # regulat fit, except M step modified (see below)
        return super(GMMHMM_impute,self).fit(obs,n_iter,thresh,params,
                                             init_params,maxrank,beamlogprob,
                                             **kwargs)
    def _compute_log_likelihood(self, obs):
        """ dont compute missing data """
        res = np.array([g.score(obs) for g in self.gmms]).T
        res[self.impute_p1:self.impute_p2,:] = 1./self.n_states
        return res

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        """
        Slight modification of the fit function, just to give us
        a reference to the posteriors, so we can modify the observations
        (impute them) in the M step.
        """
        # keep posteriors
        self.curr_posteriors = posteriors
        # regulat fit, except M step modified (see below)
        #super(GMMHMM_impute,self)._accumulate_sufficient_statistics(stats,
        #                                  obs, framelogprob,
        #                                  posteriors, fwdlattice, bwdlattice,
        #                                  params)

        # from _BaseHMM
        stats['nobs'] += 1
        if 's' in params:
            stats['start'] += posteriors[0]
        if 't' in params:
            for t in xrange(len(framelogprob)):
                zeta = (fwdlattice[t-1][:,np.newaxis] + self._log_transmat
                        + framelogprob[t] + bwdlattice[t])
                stats['trans'] += np.exp(zeta - logsum(zeta))

        # from GMMHMM
        for state,g in enumerate(self.gmms):
            gmm_logprob, gmm_posteriors = g.eval(obs)
            # don't learn from missing data!
            gmm_posteriors[self.impute_p1:self.impute_p2,:] = 1. / self.n_states
            #posteriors2 = posteriors.copy()
            #posteriors2[self.impute_p1:self.impute_p2,:] = 1. / self.n_states
            gmm_posteriors *= posteriors[:,state][:,np.newaxis]
            subops = np.concatenate([obs[:self.impute_p1,:],
                                     obs[self.impute_p2:,:]],axis=0)
            subposts = np.concatenate([gmm_posteriors[:self.impute_p1,:],
                                       gmm_posteriors[self.impute_p2:,:]],axis=0)
            tmpgmm = GMM(g.n_states, g.n_dim, cvtype=g.cvtype)
            norm = tmpgmm._do_mstep(subops,subposts,params)
            # ******************************
            stats['norm'][state] += norm
            if 'm' in params:
                stats['means'][state] += tmpgmm.means * norm[:,np.newaxis]
            if 'c' in params:
                if tmpgmm.cvtype == 'tied':
                    stats['covars'][state] += tmpgmm._covars * norm.sum()
                else:
                    cvnorm = np.copy(norm)
                    shape = np.ones(tmpgmm._covars.ndim)
                    shape[0] = np.shape(tmpgmm._covars)[0]
                    cvnorm.shape = shape
                    stats['covars'][state] += tmpgmm._covars * cvnorm

                    
    def _do_mstep(self, stats, params, usesamples=True,
                  covars_prior=1e-2, **kwargs):
        """
        Modification of the M step so we can do imputation.
        The missing data between self.impute_p1 and self.impute_p2
        has to be replaced by the most likely data.

        Slight modif, if usesamples, we sample from the GMM instead of
        using means
        """
        # call regular M step
        super(GMMHMM_impute,self)._do_mstep(stats, params, covars_prior,
                                            **kwargs)
        # modify data
        assert self.curr_obs[0].shape[1] == 12 # debug
        assert np.abs(self.curr_posteriors[10,:].sum() - 1) < 1e-10
        assert len(self.curr_obs) == 1,'imputation works with one seq at a time'
        # get state probability for each of the missing beat
        # here, gamma is in log, we want posteriors!
        # impute the beat: state prob * GMM mean for that state
        self.curr_obs[0][self.impute_p1:self.impute_p2,:] = 0.
        for t in range(self.impute_p1,self.impute_p2):
            for state,g in enumerate(self.gmms):
                if not usesamples:
                    self.curr_obs[0][t,:] += self.curr_posteriors[t,state] * g.means.sum(0)
                else:
                    self.curr_obs[0][t,:] += self.curr_posteriors[t,state] * g.rvs(1).flatten()
        # clean, insure the max and min (should be 1. and ~0.)
        maxval = max(self.curr_obs[0][:self.impute_p1,:].max(),
                     self.curr_obs[0][self.impute_p2:,:].max())
        minval = max(self.curr_obs[0][:self.impute_p1,:].min(),
                     self.curr_obs[0][self.impute_p2,:].min())
        self.curr_obs[0][np.where(self.curr_obs[0]>maxval)] = maxval
        self.curr_obs[0][np.where(self.curr_obs[0]<minval)] = minval



class Gaussian_impute(HMM.GaussianHMM):
    """
    Everything is the same, except for the computelog likelihood,
    which affects the decoding step: it ignores the missing data!
    """
    def decode(self, obs, p1, p2, maxrank=None, beamlogprob=-np.Inf):
        """
        Same as the original, except that we modify log probs for the missing
        information
        """
        obs = np.asanyarray(obs)
        framelogprob = self._compute_log_likelihood(obs)
        #*********************
        framelogprob[p1:p2,:] = 0.
        #*********************
        logprob, state_sequence = self._do_viterbi_pass(framelogprob, maxrank,
                                                        beamlogprob)
        return logprob, state_sequence

    def predict(**kwargs):
        raise NotImplementedError


def imputation(btchroma,p1,p2,nstates=3,niter=500,nstates_subseq=0,hmm=None,seed=8484):
    """
    Performs imputation
    INPUT
      btchroma        - btchroma
      p1              - first beat masked
      p2              - last beat masked + 1
      nstates         - number of HMM states
      niter           - number of iterations
      nstates_subseq  - if >0, trans matrix contains segments that are subseq themselves
                        nstates % nstates_subseq = 0 !!!
      hmm             - continue training of an hmm
      seed            - numpy random seed
    RETURN
      recon     - reconstruction, same shape as btchroma
    """
    np.random.seed(seed)
    # get the input in the right form
    obs = btchroma.copy()
    meanbeat = np.concatenate([btchroma[:,:p1],btchroma[:,p2:]],axis=1).mean()
    for p in range(p1,p2):
        obs[:,p] = meanbeat
    obs = obs.T
    means = []
    # init random gaussians
    for g in range(nstates):
        means.append(np.random.rand(12))
    # transition matrix
    if nstates_subseq == 0:
        transmat = np.ones((nstates,nstates),'float')
        startprob = np.ones(nstates,'float')/nstates
        transmat_prior = 1.
    else:
        assert nstates % nstates_subseq == 0 # subseq length must divide nstates'
        nsubseq = int(nstates / nstates_subseq)
        transmat = np.zeros((nstates,nstates),'float')
        # fill upper left blocks
        for l in range(nsubseq):
            for c in range(nsubseq):
                transmat[(l+1)*nstates_subseq-1,(c+1)*nstates_subseq-1] = 1.
        # fill upper right diags with ones
        small_trans = np.ones((nstates_subseq,nstates_subseq))
        for l in range(1,nstates_subseq):
            for c in range(l):
                small_trans[l,c] = 0.
                if c+1<l:
                    small_trans[c,l] = 0. # 2 diagonal trans, extreme
            small_trans[l,l] = 0. # super extreme, cant stay in state
        # copy small_seq
        nss = nstates_subseq
        for sub in range(nsubseq):
            transmat[sub*nss:(sub+1)*nss,
                     sub*nss:(sub+1)*nss] = small_trans.copy()
        # startprob prior
        startprob = np.zeros(nstates)
        for k in np.array(range(0,nstates,nstates_subseq)):
            startprob[k] = 1. / nsubseq
        # transmat prior
        transmat_prior = 1.
    # normalize transmat
    #transmat += np.finfo('float').eps # add eps
    for ridx in range(transmat.shape[0]):
        transmat[ridx,:] /= transmat[ridx,:].sum()
    for r in transmat:
        assert r.sum() == 1.,'bad normalization for transmat'
    transmat_zeros = np.where(transmat==0.)
    # create hmm
    if hmm is None:
        hmm = Gaussian_impute(n_states=nstates,n_dim=12,means=means,
                              startprob=startprob,transmat=transmat,
                              transmat_prior=transmat_prior)
    else:
        print 'we keep hmm passed as argument'
    seq1 = btchroma[:,:p1].T.copy()
    seq2 = btchroma[:,p2:].T.copy()
    for k in xrange(niter):
        res = hmm.fit([seq1],n_iter=1)
        res = hmm.fit([seq2],n_iter=1)
        # reset zeros of transmat
        tmptransmat = hmm.transmat
        tmptransmat[transmat_zeros] = 0.
        hmm._set_transmat(tmptransmat)
    # reconstruction
    logprob, states = hmm.decode(obs,p1,p2)
    recon1 = btchroma.copy()
    for k in range(p1,p2):
        recon1[:,k] = hmm.means[states[k]]
    # for recon2 find other parts of the song with similar state
    recon2 = btchroma.copy()
    for k in range(p1,p2):
        pos_same_state = np.where(states==states[k])[0]
        pos_same_state = filter(lambda x: x<p1 or x>=p2,pos_same_state)
        if len(pos_same_state) > 0:
            np.random.shuffle(pos_same_state)
            recon2[:,k] = recon2[:,pos_same_state[0]]
        else:
            recon2[:,k] = recon2[:,k-1] # big hack
    return recon1, recon2, hmm


