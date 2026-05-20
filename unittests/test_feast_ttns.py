import unittest
import sys
import copy
from pathlib import Path
import numpy as np
from scipy import linalg as la
from ttnsVector import TTNSVector
import basis
import mctdh_stuff
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.parseInput import parseTree, getMPS
from ttns2.contraction import TruncationEps
from ttns2.contraction import TruncationFixed
from ttns2.driver import orthogonalize
from feast import *
import util
from util_funcs import find_nearest
from magic import ipsh
from util_funcs import select_within_range


class Test_feast_ttns(unittest.TestCase):

    def setUp(self):
        EPS = 5e-9
        convTol = 1e-5
        N_STATES = 6 # also sets eigenvalue index below. sigma is 

        fOp = Path(__file__).with_name("pyr4+.op")
        Hop = mctdh_stuff.translateOperatorFile(fOp, verbose=False)

        FAC = 2
        DVRopts = [
            basis.Hermite.getOptions(N=40//FAC, HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=32//FAC, HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=16//FAC, HOx0=0, HOw=1, HOm=1),
            basis.Hermite.getOptions(N=12//FAC, HOx0=0, HOw=1, HOm=1),
        ]
        bases = [basis.electronic(2)]
        for DVRopt in DVRopts:
            bases.append(basis.basisFactory(DVRopt))

        nBas = [b.N for b in bases]
        Hop.storeMatrices(bases)
        Hop = operatornD.contractSoPOperatorSimpleUsage(Hop)
        operatornD.absorbCoeff(Hop)
        Hop.obtainMultiplyOp(bases)
        basisDict = {l:b for l,b in zip(Hop.DoFlabel, bases)}
        tns = getMPS(basisDict, 3)
        np.random.seed(13)
        #tns.setRandom(dtype=complex)
        tns.setRandom(dtype=float) # HRL suggested


        davidsonOptions = [IterativeDiagonalizationOptions(tol=1e-7, maxIter=500,maxSpaceFac=200)] * 8
        # tighter convergence 
        davidsonOptions.append(IterativeDiagonalizationOptions(tol=1e-8, maxIter=500,maxSpaceFac=200))
        # Do a loose calc with just maxD=2
        bondDimensionAdaptions = [TruncationEps(EPS, maxD=9, offset=2, truncateViaDiscardedSum=False)]
        noises = [1e-6] * 4 + [1e-7] * 4 + [1e-8] * 6
        tnsList, energies = eigenStateComputations(tns, Hop,
                                     nStates=N_STATES,
                                     nSweep=999,
                                     projectionShift=util.unit2au(9999,"cm-1"),
                                     iterativeDiagonalizationOptions=davidsonOptions,
                                     bondDimensionAdaptions= bondDimensionAdaptions,
                                     noises = noises,
                                     allowRestart=False,   
                                     saveDir=None,
                                     convTol=convTol)

        self.rmin = energies[3]*1.001
        self.rmax = energies[5]*1.001
        self.maxit = 15
        self.nc = 6  
        self.eConv = 1e-6
        self.quad = "legendre"
        self.writeOut = False


        adaptionsLinear =  [TruncationEps(EPS, maxD=5, offset=1, truncateViaDiscardedSum=False)] 
        adaptionsFitting = [TruncationEps(EPS, maxD=9, offset=1, truncateViaDiscardedSum=False)]
        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-4,maxIter=70000) 
        optionsLinear = {"nSweep":1000, "iterativeLinearSystemOptions":optsCheck,"convTol":1e-4, "verbose": True, "bondDimensionAdaptions": adaptionsLinear}
        optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":adaptionsFitting, "verbose": True}
        options = {"linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

        self.evEigh = energies
        self.uvEigh = tnsList
        self.mat = Hop
        
        m0 = 4 
        # Random orthogonal tress
        setTrees = []
        for i in range(m0):
            tns = getMPS(basisDict, 3)
            np.random.seed(20+i);tns.setRandom(dtype=complex)
            setTrees.append(tns)
        setTrees= orthogonalize(setTrees)
        
        # Make a TTNSVector list from above orthogonal trees
        guess = []
        for i in range(m0):
            guess.append(TTNSVector(setTrees[i],options))
        self.guess = guess

    
    def test_feast_ttns(self):
        evfeast, uvfeast, status = feastDiagonalization(self.mat,self.guess,
                self.nc,self.quad,self.rmin,self.rmax,self.eConv,self.maxit,
                writeOut=self.writeOut)

        typeClass = uvfeast[0].__class__
        
        with self.subTest("orthogonalization"):
            ''' Returned basis in old form is orthogonal'''
            S = typeClass.overlapMatrix(uvfeast)
            np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5)


        with self.subTest("transformationMatrix"):
            ''' XH@S@X = 1'''
            S = typeClass.overlapMatrix(uvfeast)
            assert len(uvfeast) > 1
            SmatFull = typeClass.overlapMatrix(uvfeast)
            uS = lowdinOrthoMatrix(SmatFull,status)[1]
            HmatFull = typeClass.matrixRepresentation(self.mat,uvfeast)
            uv = diagonalizeHamiltonian(uS,HmatFull)[1]
            uSH = uS@uv
            mat = uSH.T.conj()@S@uSH
            np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5)

        with self.subTest("returnType"):
            ''' Checks if the returned eigenvalue and eigenvectors are of correct type'''
            self.assertIsInstance(evfeast, np.ndarray)
            self.assertIsInstance(uvfeast, list)
            self.assertIsInstance(uvfeast[0], TTNSVector)

        with self.subTest("eigenvalue"):
            ''' Checks accuracy of the calculated eigenvalues'''

            #All contour eigenvalues
            contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
            ncontour_evs = len(contour_evs)
            nfeast_ev = len(evfeast)
            self.assertTrue((ncontour_evs <= nfeast_ev),'All eigenvalues within contour must be calculated')

            #eigenvalue accuracy:
            feast_evs = select_within_range(evfeast, self.rmin, self.rmax)[0]
            for i in range(len(contour_evs)):
                target_value = contour_evs[i]
                closest_value = find_nearest(feast_evs,target_value)[1]
                self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 4-nd decimal place')
    
        with self.subTest("eigenvector"):
            ''' Checks accuracy of the calculated eigenvectors'''

            contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
            for i in range(len(contour_evs)):
                idxE = find_nearest(self.evEigh,contour_evs[i])[0]
                idxT = find_nearest(evfeast,contour_evs[i])[0]
                options = uvfeast[0].options
                exactVector = TTNSVector(self.uvEigh[idxE],options)
                feastVector = uvfeast[idxT]

                ovlp = exactVector.vdot(feastVector)
                np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-4, err_msg = f"{ovlp=} but it should be +-1")
            
                feastVector = feastVector * np.conjugate(ovlp)
                exactVector = np.ravel(exactVector.ttns.fullTensor(canonicalOrder=True)[0])
                feastVector = np.ravel(feastVector.ttns.fullTensor(canonicalOrder=True)[0])
                np.testing.assert_allclose(exactVector,feastVector,rtol=1e-3,atol=1e-3)

if __name__ == "__main__":
    unittest.main()
