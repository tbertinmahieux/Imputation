"""
To create figures regarding different measures
"""

import os
import sys
import numpy as np
import pylab as P



# L2 - L1 - L1/2
npoints = 1000
r = np.array(range(-npoints/2,npoints/2+1,1)) * 2./npoints
y1 = r * r
y2 = np.abs(r)
y3 = np.power(np.abs(r),.5)
y4 = np.power(np.abs(r),.001)
P.figure()
P.hold(True)
P.plot(r,y1,'b-',label='L2')
P.plot(r,y2,'g-.',label='L1')
P.plot(r,y3,'r--',label='L1/2')
#P.plot(r,y4,'m-',label='L.1')
P.yticks([0.2,0.4,0.6,0.8])
P.axvspan(-.1,.1,ymin=0,ymax=1,alpha=.2,label='L0')
P.legend(loc='lower right')
P.hold(False)
P.show(mainloop=False)  # set to False for non blocking
aspect = P.gca().get_aspect()

# SQUARE WAVE
npoints = 1000
delta = .008
r = np.array(range(npoints+1)) * 1. / npoints
y1 = np.zeros(npoints+1)
y1[np.where(r<.5)] = 1. # first square
y2 = np.concatenate([y1[npoints/4:],y1[:npoints/4]])
y2[np.where(y2==0.)] = delta
y2[np.where(y2==1)] = 1-delta
y3 = np.ones(r.shape) * .5 # average
# make it longer by concat
extra = npoints/4
r = np.concatenate([r,r[:extra]+1.+1./npoints])
print r.shape
y1 = np.concatenate([y1,y1[:extra]])
y2 = np.concatenate([y2,y2[:extra]])
y3 = np.concatenate([y3,y3[:extra]])
# ........................
P.figure()
P.hold(True)
P.plot(r,y1,'b-',label='square wave')
P.plot(r,y2,'g-.',label='translated wave')
P.plot(r,y3,'r--',label='average')
P.hold(False)
P.ylim([-.1,1.1])
P.xlim([0,1.25])
P.yticks([0.2,0.4,0.6,0.8])
P.xticks([0,.5,1.])
P.legend(loc='lower right')
P.gca().set_aspect(aspect)
P.show(mainloop=True)
