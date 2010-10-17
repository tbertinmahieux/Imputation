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
#bt = 


# SMOOTH RECONSTRUCTION
bt = sio.loadmat('/home/thierry/Columbia/covers80/coversongs/covers32kENmats/metallica+Live_Shit_Binge_And_Purge_Disc_3_+09-Stone_Cold_Crazy.mp3.mat')['btchroma']
mask,p1,p2 = masking.random_patch_mask(bt,ncols=15,win=25)
recon,tmp = IMPUTATION.knn_patch(bt,mask,p1,p2,measure='eucl',win=8)
originals.append( bt[:,p1:p2].copy() )
recons.append( recon.copy() )
errs = evaluation.recon_error(bt,mask,recon)
s = 'method: NN\n'
s += 'eucl = ' + nice_nums(errs['eucl']) + '\n'
s += 'D-ENT = ' + nice_nums(errs['dent']) + '\n'
s += 'Jensen diff. = ' + nice_nums(errs['jdiff']) + '\n'
s += 'Levenshtein. = ' + nice_nums(errs['leven']) + '\n'
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
    # recon
    P.subplot(len(originals),3,k*3+2)
    P.imshow(originals[k],**pargs)
    # text
    P.subplot(len(originals),3,k*3+3)
    P.gca().set_axis_off()# remove black box around it
    P.imshow(np.zeros((10,10)),**pargs)
    P.xticks(())
    P.yticks(())
    width = 1    # out of 10
    height = 8  # out of 10
    lines = filter(lambda s: s!='',texts[k].split('\n'))
    for lid,line in enumerate(lines):
        P.text(width,height-lid, line, fontdict=None)
               #horizontalalignment='center',
               #verticalalignment='center')

P.show()

