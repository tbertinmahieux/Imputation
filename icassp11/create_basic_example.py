"""
Code to create the first figure, where we do reconstruction using a set
of methods and report some measures.
"""

import os
import sys
import numpy as np
import scipy.io as sio
import pylab as P

sys.path.append('/home/thierry/Columbia/Imputation/PythonSrc')
import evaluation
import imputation as IMPUTATION
import imputation_plca as IMPUTATION_PLCA


# load btchroma, create mask
btchroma = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/john_lennon+Double_Fantasy+05-I_m_Losing_You.mp3.mat')['btchroma']
p1=185;p2=p1+15;mask=np.ones(btchroma2.shape);mask[:,p1:p2]=0.

# method: lintrans
recon_lintrans,tmp = IMPUTATION.lintransform_patch(btchroma,mask,p1,p2,win=1)
err_lintrans = evaluation.recon_error(btchroma,mask,recon_lintrans)
# method: random
recon_rand = IMPUTATION.random_patch(btchroma,mask,p1,p2)
err_rand = evaluation.recon_error(btchroma,mask,recon_rand)
# method: average
recon_avg = IMPUTATION.average_patch(btchroma,mask,p1,p2,win=1)
err_avg = evaluation.recon_error(btchroma,mask,recon_avg)
# method: NN
recon_nn,tmp = IMPUTATION.knn_patch(btchroma,mask,p1,p2,win=8,measure='eucl')
err_nn = evaluation.recon_error(btchroma,mask,recon_nn)
# method: SIPLCA
recon_siplca = IMPUTATION_PLCA.SIPLCA_mask.analyze((btchroma*mask).copy(),
                                                   50,mask,win=40,
                                                   convergence_thresh=1e-15)[4]
err_siplca = evaluation.recon_error(btchroma,mask,recon_siplca)

# PLOT
pargs = {'origin':'lower','interpolation':'nearest','cmap':P.cm.gray_r,
         'aspect':'auto'}
pos1 = p1-10
pos2 = p2+70
P.figure()
P.hold(True)
# original and masked
P.subplot(611)
P.imshow(btchroma[:,pos1:pos2],**pargs)
P.yticks([4,8])
P.xticks()
P.subplot(612)
P.imshow((btchroma*mask)[:,pos1:pos2],**pargs)
P.yticks([4,8])
P.xticks()
# random

# nn

# avg




#%evaluation.plot_oneexample(btchroma2,mask,p1,p2,methods=['lintrans'],methods_args=[{'win':1}],measures=('eucl','kl','dent'),plotrange=(p1-10,p2+70))


P.show()
