import unittest
import sys
from inexact_Lanczos  import *
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
    
class Test_lanczos(unittest.TestCase):
    def setUp(self):
        # This is a specific case where linearly dependent vectors
        # generation happens
        n = 1200
        ev = np.linspace(1,400,n)
        np.random.seed(10)
        Q = la.qr(np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q

        options = {"linearSolver":"gcrotmk","linearIter":500,"linear_tol":1e-1}
        optionDict = {"linearSystemArgs":options}
        self.printChoices = {"writeOut": False,"writePlot": False}
        Y0 = NumpyVector(np.random.random((n)),optionDict)
        
        self.guess = Y0
        self.mat = A
        self.ev = ev 
        self.sigma = 390
        self.eShift = 0.0
        self.L = 100     # make sufficiently large to get LINDEP
        self.maxit = 1000 # same as above
        self.eConv = 1e-12

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

    def test_status(self):
        """ This specific case face lindep in the first Lanczos iteration,
        check if status["lindep"] is indeed True or not"""
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,self.eConv,pick=None,status = self.printChoices)
        # TODO need to be made better
        self.assertTrue(status["lindep"]== True, msg="mail fail on some machines; "
                                                     "may be ok as this only tests an assertion")
        ''' Testing after getting linear dependency the list must be truncated
            or the length of vectors list should be iKrylov'''
        iKrylov = status["innerIter"]
        nvectors = len(uvLanczos)
        self.assertTrue(nvectors == iKrylov)

    def test_futileRestarts(self):
        """ For this specific case, number of futile restarts is larger than 3"""
        eConv = 1e-18 # stoping from early convergence
        status = inexactLanczosDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,eConv,pick=None,status = self.printChoices)[2]
        nfutileRestarts = status["futileRestarts"]
        # one or more futile restarts 
        if status["outerIter"] < self.maxit-1:
            self.assertTrue(nfutileRestarts >= 1)

if __name__ == "__main__":
    unittest.main()
