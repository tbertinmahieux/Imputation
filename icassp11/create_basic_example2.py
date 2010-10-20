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

def errs_to_str(errs):
    s = 'eucl. = ' + nice_nums(errs['eucl'])
    s += ', delta diff. = ' + nice_nums(errs['ddiff'])
    s += ', D-ENT = ' + nice_nums(errs['dent'])
    return s

# load btchroma, create mask
btchroma = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/a_ha+Hunting_High_and_Low+01-Take_On_Me.mp3.mat')['btchroma']
p1=290;p2=p1+20;mask=np.ones(btchroma.shape);mask[:,p1:p2]=0.

# method: lintrans
recon_lintrans,tmp = IMPUTATION.lintransform_patch(btchroma,mask,p1,p2,win=1)
err_lintrans = evaluation.recon_error(btchroma,mask,recon_lintrans)
# method: random
#recon_rand = IMPUTATION.random_patch(btchroma,mask,p1,p2)
#err_rand = evaluation.recon_error(btchroma,mask,recon_rand)
# method: average
recon_avg = IMPUTATION.average_patch(btchroma,mask,p1,p2,win=1)
err_avg = evaluation.recon_error(btchroma,mask,recon_avg)
# method: NN
recon_nn,tmp = IMPUTATION.knn_patch(btchroma,mask,p1,p2,win=11,measure='eucl')
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
pos2 = p2+10
P.figure()
P.hold(True)
# original and masked
P.subplot(611)
P.imshow(btchroma[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([])
P.title('ORIGINAL',fontsize='small')
P.subplot(612)
P.imshow((btchroma*mask)[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([])
P.title('MASKED',fontsize='small')
# random
#P.subplot(713)
#P.imshow(recon_rand[:,pos1:pos2],**pargs)
#P.gca().grid(False)
#P.yticks([4,8])
#P.xticks([])
#P.title('RANDOM, ' + errs_to_str(err_rand),fontsize='small')
# nn
P.subplot(613)
P.imshow(recon_nn[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([])
P.title('1NN, ' + errs_to_str(err_nn),fontsize='small')
# siplca
P.subplot(614)
P.imshow(recon_siplca[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([])
P.title('SIPLCA, ' + errs_to_str(err_siplca),fontsize='small')
# avg
P.subplot(615)
P.imshow(recon_avg[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([])
P.title('AVERAGE, ' + errs_to_str(err_avg),fontsize='small')
# lintrans
P.subplot(616)
P.imshow(recon_lintrans[:,pos1:pos2],**pargs)
P.gca().grid(False)
P.yticks([])
P.xticks([0,20,39],fontsize='small')
#P.xticks([])
P.title('LIN. TRANS., ' + errs_to_str(err_lintrans),fontsize='small')


P.show()
