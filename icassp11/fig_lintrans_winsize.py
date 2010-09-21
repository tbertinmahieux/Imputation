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
    winsize = [1,2,3,4]
    eucl = [0.0411784244896,0.0370506783561,
            0.0340551163455,0.0313302079853]
    eucl_std = [0.0276543117112,0.0258182303757,
                0.0242988658679,0.0232631365607]
    kl = [0.115849617038,0.106086651178,
          0.0989381442861,0.0923258474298]
    kl_std = [0.127330651343,0.126658861034,
              0.107594782082,0.107594782082]
    # plot
    P.figure()
    l1 = P.plot(winsize,eucl,label='euclidean')
    P.ylabel('euclidean distance')
    ax2 = P.twinx() # second axis
    l2 = P.plot(winsize,kl,'--')
    P.ylabel('KL divergence')
    P.xticks( winsize )
    P.xlabel('window size')
    P.legend( (l1,l2) , ('euclidean','KL') )
    P.show()
