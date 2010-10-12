"""
Generate curve of divergence for the linear transform based
on the window size

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np
import pylab as P



if __name__ == '__main__':

    # data
    winsize = [1,2,3,4,5,6]
    eucl = [0.0449717189645,0.0441956895732,
            0.044850135746,0.517288678962,
            0.0772906118926,0.0791333591867]
    eucl_std = [0.030589862504,0.032154520332,
                0.0401470970956,97.9208346678,
                3.37503756986,2.23124618451]
    kl = [0.125879685737,0.125569492093,
          0.537071280374,0.139147344057,
          0.160516193591,0.193887723751]
    kl_std = [0.136761459576,0.161251003088,
              79.9330029095,1.1232832439,
              2.45652263388,5.13061036956]
    # plot
    P.figure()
    l1 = P.plot(winsize,eucl,label='euclidean')
    P.ylabel('euclidean distance')
    P.xlabel('window size')
    ax2 = P.twinx() # second axis
    l2 = P.plot(winsize,kl,'--')
    P.ylabel('KL divergence')
    P.xticks( winsize )
    P.legend( (l1,l2) , ('euclidean','KL') )
    P.show()


