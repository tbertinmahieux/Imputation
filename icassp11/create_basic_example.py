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


def nice_nums(val):
    return evaluation.nice_nums(val)

# load btchroma, create mask
btchroma = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/john_lennon+Double_Fantasy+05-I_m_Losing_You.mp3.mat')['btchroma']
p1=185;p2=p1+15;mask=np.ones(btchroma.shape);mask[:,p1:p2]=0.

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
P.subplot(711)
P.imshow(btchroma[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
P.subplot(712)
P.imshow((btchroma*mask)[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
# random
P.subplot(713)
P.imshow(recon_rand[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
P.title('allo')
# nn
P.subplot(714)
P.imshow(recon_nn[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
# avg
P.subplot(715)
P.imshow(recon_avg[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
# siplca
P.subplot(716)
P.imshow(recon_siplca[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
P.xticks([])
# lintran
P.subplot(717)
P.imshow(recon_lintrans[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([4,8])
#P.xticks([])  we let the last one!

#%evaluation.plot_oneexample(btchroma2,mask,p1,p2,methods=['lintrans'],methods_args=[{'win':1}],measures=('eucl','kl','dent'),plotrange=(p1-10,p2+70))


P.show()
