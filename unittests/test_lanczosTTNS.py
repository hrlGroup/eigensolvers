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
import operatornD, operator1D
from util_funcs import calculateTarget

class Test_lanczos(unittest.TestCase):
    def setUp(self):
        dtype = float  # another option to try: complex
        basisDict = {
            "q0": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q1": basis.SincAB(basis.SincAB.getOptions(N=2)),
            "q2": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q3": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "q4": basis.SincAB(basis.SincAB.getOptions(N=3)),
            "z5": basis.SincAB(basis.SincAB.getOptions(N=5)),
        }
        treeString = """
        0> 3 3
            1> 4 [q0]
                2> [q1 q2]
            1> 2 3 4
                2> [q3]
                2> [q4]
                2> [z5]
                """
        ttns = parseTree(treeString, basisDict, returnType="TTNS",dtype=dtype)
        np.random.seed(1212)
        ttns.setRandom()

        bases = list(basisDict.values())
        labels = list(basisDict.keys())
        nBas = [b.N for b in bases]
        Hop = operatornD.operatorSumOfProduct(nDim=len(nBas), nSum=3, DoFlabel=labels)
        for iSum in range(Hop.nSum-1): # have one unit term
            for iDim in range(Hop.nDim):
                Hop[iDim, iSum] = operator1D.general(str=f"{iDim}_{iSum}")
                if dtype == complex:
                    Hop[iDim, iSum].mat = npu.randomComplexHermitian(nBas[iDim])
                else:
                    Hop[iDim, iSum].mat = npu.randomSymmetric(nBas[iDim])
        Hop.obtainMultiplyOp(nBases=nBas)
        H = Hop.toFULLMatrix(nBas)
        assert npu.isHermitian(H)
        
        self.mat = Hop
        evEigh, uvEigh = la.eigh(H)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
        
        MAX_D = 100 
        EPS = 5e-9
        bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
        nsweepOrtho = 800
        orthoTol = 1e-08
        optShift = 0.0

        siteLinearTol = 1e-3
        globalLinearTol = 1e-2
        nsweepLinear = 1000

        fittingTol = 1e-9
        nsweepFitting = 1000

        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol,maxIter=2000) 
        optionsOrtho = {"nSweep":nsweepOrtho, "convTol":orthoTol, "optShift":optShift, "bondDimensionAdaptions":bondDimensionAdaptions}
        optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,"convTol":globalLinearTol,"bondDimensionAdaptions":bondDimensionAdaptions}
        optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,"bondDimensionAdaptions":bondDimensionAdaptions}
        options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

        tns = TTNSVector(ttns,options)
        self.guess = tns
        self.eShift = 0.0
        self.maxit = 20
        self.L = 30
        self.eConv = 1e-7
        self.writeOut = False
        self.convertUnit = "cm-1"

    def test_LanczosTTNS(self):
        target = calculateTarget(self.evEigh,4)
        sigma = target + self.eShift
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,sigma,self.L,
                                                  self.maxit,self.eConv,writeOut=self.writeOut,eShift=self.eShift,
                                                  convertUnit=self.convertUnit,pick=None)
        typeClass = uvLanczos[0].__class__
        self.assertTrue(len(uvLanczos) > 1)
        Sfull = typeClass.overlapMatrix(uvLanczos)
        with self.subTest("returnType"):
            '''Check that the returned eigenvalues and eigenvectors have the correct types.'''
            self.assertIsInstance(evLanczos, np.ndarray)
            self.assertIsInstance(uvLanczos, list)
            self.assertIsInstance(uvLanczos[0], TTNSVector)
        with self.subTest("orthogonal"):
            '''Check that the returned basis is orthogonal.'''
            np.testing.assert_allclose(Sfull,np.eye(Sfull.shape[0]),atol=1e-5)
        with self.subTest("extension"):
            '''Check that matrix extension works.'''
            S1 = typeClass.overlapMatrix(uvLanczos[:-1])
            S = typeClass.extendOverlapMatrix(uvLanczos,S1)
            qtAqfull = typeClass.matrixRepresentation(self.mat,uvLanczos)
            qtAq1 = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
            qtAq = typeClass.extendMatrixRepresentation(self.mat,uvLanczos,qtAq1)
            np.testing.assert_allclose(S,Sfull,atol=1e-9)
            np.testing.assert_allclose(qtAq,qtAqfull,atol=1e-9)

    def test_eigenvaluesAndVectors(self):
        places = [4,8,12,16]
        for p in places:
            target = calculateTarget(self.evEigh,p)
            sigma = target + self.eShift
            evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,sigma,self.L,
                    self.maxit,self.eConv,writeOut=self.writeOut,eShift=self.eShift,
                    convertUnit=self.convertUnit,pick=None)
            target_value = find_nearest(evLanczos,sigma)[1]
            closest_value = find_nearest(self.evEigh,sigma)[1]
            with self.subTest("eigenvalue"):
                '''Check that relative eigenvalue errors are at most 1e-5.'''
                relError = abs(target_value-closest_value)/abs(closest_value)
                self.assertTrue((relError <= 1e-5),'Relative accuracy with respect to exact levels is higher than 1e-5')
            with self.subTest("eigenvector"):
                '''Check that the calculated eigenvector is accurate up to 5e-4.
                The previous test ensures relative eigenvalue errors <= 1e-2.'''
                idxE = find_nearest(self.evEigh,sigma)[0]
                idxT = find_nearest(evLanczos,sigma)[0]
                exactUV = self.uvEigh[:,idxE]
                ttnsT = np.ravel(uvLanczos[idxT].ttns.fullTensor(canonicalOrder=True)[0])
                ovlp = np.vdot(ttnsT,exactUV)
                np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg = f"{ovlp=} but it should be +-1")
                lanczosTree = ttnsT* ovlp
                np.testing.assert_allclose(exactUV,lanczosTree,rtol=8e-3,atol=5e-4)

if __name__ == '__main__':
    unittest.main()
