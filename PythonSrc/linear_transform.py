"""
Code to learn a linear transform using gradient descent.

T. Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu
"""

import os
import sys
import numpy as np



class LinTrans():
    """
    Class that implements a linear trnsform.
    """
    def __init__(self,inputlen,outputlen):
        """
        Constructor
        INPUT
           inputlen     - length of the input data 
           outputlen    - length of the output data
        """
        self._inputlen = inputlen
        self._outputlen = outputlen
        self.__inittransform__()

    def __inittransform__(self):
        """
        Initialize the matrix randomly
        """
        #self._transform = np.random.rand(self._inputlen,
        #                                 self._outputlen)
        #self._bias = np.random.rand(1,self._outputlen)
        self._transform = np.ones((self._inputlen,self._outputlen)) / self._inputlen
        self._bias = np.zeros((1,self._outputlen))

    def __computeerrors__(self,datain,dataout):
        """
        Compute prediction error for a list of input/output
        INPUT
           datain    - array of inputs
           dataout   - array of outputs
        """
        return map(lambda x: self.__computeerror__(datain[x],dataout[x]),
                   range(len(datain)))

    def __computeerror__(self,datain,dataout):
        """
        Compute prediction error for one pair of input/output
        Uses euclidean distance.
        INPUT
           datain    - one input
           dataout   - one output
        """
        return np.square(self.predicts(datain)-dataout).sum()

    def __computegrads__(self,datain,dataout):
        """
        Compute gradient for a list of input/output
        INPUT
           datain    - array of inputs
           dataout   - array of outputs
        """
        return map(lambda x: self.__computegrad__(datain[x],dataout[x]),
                   range(len(datain)))

    def __computegrad__(self,datain,dataout):
        """
        Compute gradient for one pair of input/output
        Based on euclidean distance error function
        INPUT
           datain    - array of inputs
           dataout   - array of outputs
        """
        res = 2 * (self.predicts(datain) - dataout.reshape(1,self._outputlen))
        return res

    def train(self,datain,dataout,lrate=.1,niter=100):
        """
        Train the transform from a list of data input/output
        using batch method
        INPUT
           datain    - array of inputs
           dataout   - array of outputs
           lrate     - earning rate, default=.1
           niter     - number of iteration on the full dataset, niter=100
        """
        for it in xrange(niter):
            # compute gradients
            grads = self.__computegrads__(datain,dataout)
            # transform gradients
            transgrads = [grads[i] * datain[i] for i in range(len(datain))]
            batchtransgrad = np.array(transgrads).mean(axis=0)
            # update transform
            self._transform -= lrate * batchtransgrad
            # bias gradient
            batchbiasgrad = np.array(grads).mean(axis=0)
            # update bias
            self._bias -= lrate * batchbiasgrad.reshape(self._bias.shape)
        
    def predicts_all(self,datain):
        """
        Apply the linear transform to an array of input
        """
        return map(lambda x: self.predicts(x),datain)

    def predicts(self,datain):
        """
        Apply the linear transform on one input
        """
        #return datain.reshape(1,self._inputlen) * self._transform
        return np.dot(datain.reshape(1,self._inputlen) , self._transform) + self._bias




def die_with_usage():
    """
    HELP MENU
    """
    print 'Simple linear transform library,'
    print 'run: python linear_transform.py -go <lrate> <niter>'
    print 'to launch a debugging / demo run'
    sys.exit(0)


if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    # DEBUG / DEMO
    lrate = .1
    niter = 100
    if len(sys.argv) > 2:
        lrate = float(sys.argv[2])
    if len(sys.argv) > 3:
        niter = int(sys.argv[3])
    inlen = 5
    outlen = 5
    ndata = 100
    # create a transform
    trans = np.random.rand(inlen,outlen)
    # create data
    indata = map(lambda x: np.random.rand(inlen,1),range(ndata))
    outdata = map(lambda x: np.dot(x.T , trans),indata)
    # init linear transform
    lintrans = LinTrans(inlen,outlen)
    # transform error
    print 'matrix dist = ',np.square(trans-lintrans._transform).sum()
    # train
    lintrans.train(indata,outdata,lrate=lrate,niter=niter)
    # transform error
    print 'matrix dist = ',np.square(trans-lintrans._transform).sum()
    
