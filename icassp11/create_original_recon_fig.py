"""
Code to create a 3 column figure
first original, secon reconstruction
third text with error measures

Super specific to my computer! - TBM
"""


import os
import sys
import numpy as np
import scipy.io as sio
import pylab as P

sys.path.append('/home/thierry/Columbia/Imputation/PythonSrc')
import masking
import evaluation
import imputation as IMPUTATION

originals = []
recons = []
texts = []



def nice_nums(val):
    return evaluation.nice_nums(val)


# EASY RECONSTRUCTION
bt = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/beatles+Revolver+14-Tomorrow_Never_Knows.mp3.mat')['btchroma']
mask,p1,p2 = masking.random_patch_mask(bt,ncols=15,win=25)
recon,tmp = IMPUTATION.lintransform_patch(bt,mask,p1,p2,win=1)
errs = evaluation.recon_error(bt,mask,recon)
s = 'method: linear transform\n'
s += 'eucl = ' + nice_nums(errs['eucl']) + '\n'
s += 'd1/2 = ' + nice_nums(errs['lhalf']) + '\n'
s += 'D-ENT = ' + nice_nums(errs['dent']) + '\n'
s += 'Jensen diff. = ' + nice_nums(errs['jdiff']) + '\n'
s += 'Levenshtein. = ' + nice_nums(errs['leven']) + '\n'
originals.append( bt[:,p1:p2].copy() )
recons.append( recon[:,p1:p2].copy() )
texts.append(s)


# SMOOTH RECONSTRUCTION
bt = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/metallica+Live_Shit_Binge_And_Purge_Disc_3_+09-Stone_Cold_Crazy.mp3.mat')['btchroma']
mask,p1,p2 = masking.random_patch_mask(bt,ncols=15,win=25)
recon,tmp = IMPUTATION.knn_patch(bt,mask,p1,p2,measure='eucl',win=8)
errs = evaluation.recon_error(bt,mask,recon)
s = 'method: NN\n'
s += 'eucl = ' + nice_nums(errs['eucl']) + '\n'
s += 'd1/2 = ' + nice_nums(errs['lhalf']) + '\n'
s += 'D-ENT = ' + nice_nums(errs['dent']) + '\n'
s += 'Jensen diff. = ' + nice_nums(errs['jdiff']) + '\n'
s += 'Levenshtein. = ' + nice_nums(errs['leven']) + '\n'
originals.append( bt[:,p1:p2].copy() )
recons.append( recon[:,p1:p2].copy() )
texts.append(s)


# RANDOM RECON
bt = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32k/Caroline_No/beach_boys+Pet_Sounds+13-Caroline_No.mp3.mat')['btchroma']
mask,p1,p2 = masking.random_patch_mask(bt,ncols=15,win=25)
recon = IMPUTATION.random_patch(bt,mask,p1,p2)
errs = evaluation.recon_error(bt,mask,recon)
s = 'method: Random\n'
s += 'eucl = ' + nice_nums(errs['eucl']) + '\n'
s += 'd1/2 = ' + nice_nums(errs['lhalf']) + '\n'
s += 'D-ENT = ' + nice_nums(errs['dent']) + '\n'
s += 'Jensen diff. = ' + nice_nums(errs['jdiff']) + '\n'
s += 'Levenshtein. = ' + nice_nums(errs['leven']) + '\n'
originals.append( bt[:,p1:p2].copy() )
recons.append( recon[:,p1:p2].copy() )
texts.append(s)



# done, plot
assert len(originals) == len(recons)
assert len(originals) == len(texts)
print 'we have '+str(len(originals))+' pairs original/recon'


P.figure()
P.hold(True)
pargs = {'interpolation':'nearest','aspect':'auto',
         'cmap':P.cm.gray_r,'origin':'lower'}
for k in range(len(originals)):
    # original
    P.subplot(len(originals),3,k*3+1)
    P.imshow(originals[k],**pargs)
    if k + 1 < len(originals):
        P.xticks(())
    # recon
    P.subplot(len(recons),3,k*3+2)
    P.imshow(recons[k],**pargs)
    if k + 1 < len(originals):
        P.xticks(())
    P.yticks(())
    # text
    P.subplot(len(originals),3,k*3+3)
    P.gca().set_axis_off()# remove black box around it
    P.imshow(np.zeros((10,10)),**pargs)
    P.xticks(())
    P.yticks(())
    width = 1    # out of 10
    height = -1  # out of 10
    lines = filter(lambda s: s!='',texts[k].split('\n'))
    P.text(width,height, texts[k], fontdict=None)


P.show()

