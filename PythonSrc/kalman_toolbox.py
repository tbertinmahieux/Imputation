"""
Wrapper around Murphy's kalman toolbox

Matlab required.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import tempfile
import numpy as np
# MATLAB wrapper, we remove the dperecation warnings
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
from mlabwrap import mlab
warnings.filterwarnings('default',category=DeprecationWarning)
# kalman toolbox path
kal_toolbox_path = '/home/thierry/Columbia/Imputation/PythonSrc/KalmanAll'
mlab.warning('off','all') # hack, remove warnings
                          # some functions are redefined but who cares!
mlab.addpath(kal_toolbox_path)
mlab.addpath(os.path.join(kal_toolbox_path,'Kalman'))
mlab.addpath(os.path.join(kal_toolbox_path,'KPMstats'))
mlab.addpath(os.path.join(kal_toolbox_path,'KPMtools'))
mlab.warning('on','all')


def learn_kalman(data,A,C,Q,R,initx,initV,niter,diagQ=1,diagR=1):
    """
    Main function, take initial parameters and train a Kalman filter.
    We assume a model:
      x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
      y(t)   = C*x(t) + v(t),  v ~ N(0, R)
    INPUT
      data   - one sequence, one observation per col
      A      - DxD matrix, D dimension of hidden state
      C      - 12xD matrix (12 for chromas)
      Q      - covariance matrix, for instance rand(D,D) . rand(D,D)'
      R      - covariance matrix, for instance rand(12,12) . rand(12,12)'
      initx  - Dx1 matrix, init hidden state
      initV  - covariance matrix, for instance rand(D,D) . rand(D,D)'
      niter  - max number of iterations
      diagQ  - if 1, Q is diagonal (set to 0 otherwise)
      diagR  - if 1, R is diagonal (set to 0 otherwise)
    RETURN
      A
      C
      Q
      R
      initx
      initv
      LL     - log likelihood
    """
    res = mlab.learn_kalman(data, A, C, Q, R, initx, initV, niter,
                            diagQ, diagR, nout=7)
    #A, C, Q, R, INITX, INITV, LL = res
    return res



def learn_kalman_missing_patch(data,p1,p2,A,C,Q,R,initx,initV,niter,
                               diagQ=1,diagR=1):
    """
    Main function, take initial parameters and train a Kalman filter.
    We assume a model:
      x(t+1) = A*x(t) + w(t),  w ~ N(0, Q),  x(0) ~ N(init_x, init_V)
      y(t)   = C*x(t) + v(t),  v ~ N(0, R)
    INPUT
      data   - one sequence, one observation per col
      A      - DxD matrix, D dimension of hidden state
      C      - 12xD matrix (12 for chromas)
      Q      - covariance matrix, for instance rand(D,D) . rand(D,D)'
      R      - covariance matrix, for instance rand(12,12) . rand(12,12)'
      initx  - Dx1 matrix, init hidden state
      initV  - covariance matrix, for instance rand(D,D) . rand(D,D)'
      niter  - max number of iterations
      diagQ  - if 1, Q is diagonal (set to 0 otherwise)
      diagR  - if 1, R is diagonal (set to 0 otherwise)
    RETURN
      A
      C
      Q
      R
      initx
      initv
      LL     - log likelihood
    """
    res = mlab.learn_kalman_missing_patch(data, p1+1, p2, A, C, Q, R,
                                          initx, initV, niter,
                                          diagQ, diagR, nout=8)
    #recon A, C, Q, R, INITX, INITV, LL = res
    return res


def imputation(data,p1,p2,dimstates=10,diagQ=1,diagR=1,niter=20,seed=8484):
    """
    Does imputation y initializing and training a kalman filter.
    INPUT
       data      - btchroma
       p1        - masked patch in cols data[:,p1:p2]
       p2        -            " " 
       dimstates - dimension of hidden states
       diagQ     - if 1, diagonal covariance for states
       diagR     - if 1, diagonal covariance for outputs
       niter     - number of iterations
       seed      - seed for numpy random
    RETURN
       recon     - reconstruction, same size as data
    """
    np.random.seed(seed)
    # redirect printouts!
    tmpouts = tempfile.TemporaryFile(mode='w')
    sys.stdout = tmpouts
    try:
        # init matrices
        D = dimstates # for convenience
        nc = data.shape[0] # number of chromas
        #A = np.eye(D) * np.random.rand(1,D) / np.sqrt(D)
        A = np.ones((D,D),'float') / (D * D)
        C = np.random.rand(nc,D)
        Q = map(lambda x: np.dot(x.T,x),[np.random.rand(D,D)])[0] / np.sqrt(D)
        #R = map(lambda x: np.dot(x.T,x),[np.random.rand(nc,nc)])[0] / np.sqrt(nc)
        # R starts with the actual covariance of the data
        #R = np.cov(np.concatenate([data[:,:p1],data[:,:p2]],axis=1))
        R = np.ones((nc,nc),'float') / (nc * nc)
        if diagQ == 1:
            Q = np.eye(D) * Q.diagonal()
        if diagR == 1:
            R = np.eye(nc) * R.diagonal()
        initx = np.random.rand(D,1) / np.sqrt(D)
        initV = np.eye(D)
        # launch
        res = learn_kalman_missing_patch(data,p1,p2,A,C,Q,R,initx,initV,niter,
                                         diagQ=diagQ,diagR=diagR)
        # fine tune recon
        recon = res[0]
        recon[:,:p1] = data[:,:p1]
        recon[:,p2:] = data[:,p2:]
        assert not np.isnan(recon).any(),'recon has NaN'
        assert not np.isinf(recon).any(),'recon has inf'
        recon[np.where(recon>1.)] = 1.
        recon[np.where(recon<0.)] = 0.
    except:
        # problem, print the outputs we hid
        tmpouts.seek(0) # get back to the beginning
        print tmpouts.xreadlines()
        raise
    finally:
        # reset output
        tmpouts.close()
        sys.stdout = sys.__stdout__
    # done
    return recon

    

if __name__ == '__main__':

    print 'experiment in missing data'
    import evaluation
    import masking
    import numpy as np
    import scipy.io as sio

    niter = 100
    D = 100
    if len(sys.argv) > 1:
        niter = int(sys.argv[1])
        print 'niter =',niter
    if len(sys.argv) > 2:
        D = int(sys.argv[2])
        print 'D =',D
    
    btchroma = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32k/Caroline_No/beach_boys+Pet_Sounds+13-Caroline_No.mp3.mat')['btchroma']

    mask,p1,p2 = masking.random_patch_mask(btchroma,ncols=10)

    # init data
    np.random.seed(8484)
    A = np.eye(D) * np.random.rand(1,D) / np.sqrt(D)
    C = np.random.rand(12,D)
    Q = map(lambda x: np.dot(x.T,x),[np.random.rand(D,D)])[0] / np.sqrt(D)
    R = map(lambda x: np.dot(x.T,x),[np.random.rand(12,12)])[0] / np.sqrt(12)
    initx = np.random.rand(D,1) / np.sqrt(D)
    initV = np.eye(D)

    # go
    res = learn_kalman_missing_patch(btchroma*mask,p1,p2,A,C,Q,R,initx,initV,
                                     niter,diagQ=1,diagR=1)
    recon = res[0]
    recon[:,:p1] = btchroma[:,:p1]
    recon[:,p2:] = btchroma[:,p2:]
    assert not np.isnan(recon).any(),'recon has NaN'
    assert not np.isinf(recon).any(),'recon has inf'
    recon[np.where(recon>1.)] = 1.
    recon[np.where(recon<0.)] = 0.


    # analysis
    print 'recon shape:',recon.shape
    div_eucl = evaluation.recon_error(btchroma,mask,recon,measure='eucl')
    div_kl = evaluation.recon_error(btchroma,mask,recon,measure='kl')
    print 'eucl div:',div_eucl
    print 'kl div:',div_kl
    
    # comparison
    import imputation as IMPUTATION
    print 'average, win=3'
    recon = IMPUTATION.average_patch(btchroma,mask,p1,p2,win=3)
    div_eucl = evaluation.recon_error(btchroma,mask,recon,measure='eucl')
    div_kl = evaluation.recon_error(btchroma,mask,recon,measure='kl')
    print 'eucl div:',div_eucl
    print 'kl div:',div_kl
    print 'kkn eucl:'
    recon,used_cols = IMPUTATION.knn_patch(btchroma,mask,p1,p2,win=7,measure='eucl')
    div_eucl = evaluation.recon_error(btchroma,mask,recon,measure='eucl')
    div_kl = evaluation.recon_error(btchroma,mask,recon,measure='kl')
    print 'eucl div:',div_eucl
    print 'kl div:',div_kl
    print 'linear transform'
    recon,proj = IMPUTATION.lintransform_patch(btchroma,mask,p1,p2,win=2)
    div_eucl = evaluation.recon_error(btchroma,mask,recon,measure='eucl')
    div_kl = evaluation.recon_error(btchroma,mask,recon,measure='kl')
    print 'eucl div:',div_eucl
    print 'kl div:',div_kl



