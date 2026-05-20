import unittest
import sys
from inexact_Lanczos  import (lowdinOrthoMatrix,diagonalizeHamiltonian,
        basisTransformation,inexactLanczosDiagonalization)
import numpy as np
from scipy import linalg as la
from ttnsVector import TTNSVector
from util_funcs import find_nearest
import time
import basis
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from util import npu
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import orthogonalize
import operatornD, operator1D
from util_funcs import calculateTarget

class Test_lanczos(unittest.TestCase):
    def setUp(self):
        k = 0.2
        Hop = operatornD.operatorSumOfProduct(nDim=2, nSum=5, DoFlabel="q0 q1".split())
        for s in range(2):
            Hop[s,s] = operator1D.KE(1)
        Hop[0,2] = operator1D.fx(lambda x:.5 * x**4)
        Hop[1,3] = operator1D.fx(lambda x:.5 * x**4)
        Hop[0,4] = operator1D.fx(lambda x,k=k: -k*x**2)
        Hop[1,4] = operator1D.fx(lambda x: x**2)
        N = 64
        rBas = basis.Hermite
        basisOpt =  rBas.getOptions(N=N, HOx0=0,HOm=1,HOw=1)
        bases = []
        for b in [1,2]:
            bases.append(basis.basisFactory(basisOpt))
        Hop.storeMatrices(bases)
        if False:
            Ns = [b.N for b in bases]
            H = Hop.toFULLMatrix(Ns)
            self.evRef = la.eigvalsh(H,subset_by_index=[8,9])
            del H
        else:
            self.evRef = [6.236111234855307] * 2
        self.sigma = 6.2 # 6.0 is much more difficult
        basisDict = {
            "q0": bases[0],
            "q1": bases[1]
        }
        treeString = """
        0> 4 [q0]
            1> [q1]
                """
        np.random.seed(1212)
        ttns1 = parseTree(treeString, basisDict, returnType="TTNS")
        ttns1.setRandom()
        ttns2 = parseTree(treeString, basisDict, returnType="TTNS")
        ttns2.setRandom()

        MAX_D = 22
        EPS = 1e-9
        bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
        ttns1, ttns2 = orthogonalize([ttns1,ttns2], bondDimensionAdaptions=bondDimensionAdaptions)
        bondDimensionAdaptions = None
        bondDimensionAdaptionsOrtho = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
        bondDimensionAdaptionsFit = [TruncationEps(EPS/20, maxD=MAX_D*2, offset=2, truncateViaDiscardedSum=False)]
        nsweepOrtho = 40
        orthoTol = 1e-06
        optShift = 0.0

        siteLinearTol = 1e-3
        globalLinearTol = 1e-2
        nsweepLinear = 20

        fittingTol = 1e-9
        nsweepFitting = 20

        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol,maxIter=2000) 
        optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift,
                        "bondDimensionAdaptions":bondDimensionAdaptionsOrtho}
        optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,
                         "convTol":globalLinearTol,"bondDimensionAdaptions":bondDimensionAdaptions}
        optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,
                          "bondDimensionAdaptions":bondDimensionAdaptionsFit}
        options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear,
                   "stateFittingArgs":optionsFitting}

        tns1 = TTNSVector(ttns1,options)
        tns2 = TTNSVector(ttns2,options)
        self.Hop = Hop
        self.guess = [tns1,tns2]
        self.nBlock = 2
        self.eShift = 0.0
        self.maxit = 20
        self.L = 10
        self.eConv = 1e-8
        self.writeOut = False

    def test_LanczosTTNSBlock(self):
        nBlock = self.nBlock
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.Hop,self.guess,self.sigma,self.L,
                                                  self.maxit,self.eConv,writeOut=self.writeOut)
        # Difficult case!
        np.testing.assert_allclose(evLanczos[:nBlock], self.evRef, rtol=self.eConv*8, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
