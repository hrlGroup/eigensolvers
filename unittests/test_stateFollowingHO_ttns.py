import unittest
import sys
import copy
from pathlib import Path
import numpy as np
from scipy import linalg as la
from ttnsVector import TTNSVector
import basis
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp
import mctdh_stuff
import basis
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree, getMPS
from ttns2.contraction import TruncationEps
from inexact_Lanczos import inexactLanczosDiagonalization
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.contraction import TruncationFixed
import util
from util_funcs import find_nearest


class Test_stateFollowing(unittest.TestCase):

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
        tns.setRandom()
        #tns.toPdf()


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
        bondDimensionAdaptionsFitting = [TruncationEps(EPS, maxD=5, offset=1, truncateViaDiscardedSum=False)]
        bondDimensionAdaptionsLinear =  [TruncationEps(EPS, maxD=5, offset=1, truncateViaDiscardedSum=False)] 

        maxit = 10
        L = 6
        eConv = 1e-6 
        idx = N_STATES-2  # states 1,2 and 3,4 are degenerate
        target = energies[idx] * 1.001 # making sure it is not an eigenvalue

        siteLinearTol = 1e-3
        globalLinearTol = 1e-2
        nsweepLinear = 1000


        fittingTol = 1e-9
        nsweepFitting = 1000

        optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=siteLinearTol,maxIter=70000) 
        verbose = False
        optionsOrtho = {} # not used
        optionsLinear = {"nSweep":nsweepLinear, "iterativeLinearSystemOptions":optsCheck,"convTol":globalLinearTol, "verbose": verbose, "bondDimensionAdaptions": bondDimensionAdaptionsLinear}
        optionsFitting = {"nSweep":nsweepFitting, "convTol":fittingTol,"bondDimensionAdaptions":bondDimensionAdaptionsFitting, "noises":[1e-6]*4}
        #options = {"linearSystemArgs":optionsLinear}
        options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

        self.writeOut = False
        ovlpRef = TTNSVector(tnsList[idx+1],options)
        self.energyRef = energies[idx+1]
        self.pick = get_pick_function_maxOvlp(ovlpRef)
        
        self.target = target
        self.evEigh = energies
        self.uvEigh = tnsList
        # Note: the guess needs to be "close" to the reference
        # Here: truncate it
        guess = copy.deepcopy(ovlpRef.ttns)
        #guess.setRandom() 
        guess.adaptBondDimension(truncation=TruncationFixed(1))
        self.guess = TTNSVector(guess,options)
        self.mat = Hop
        self.L = L
        self.eConv = eConv
        self.maxit = maxit
        self.ovlpRef = ovlpRef

    def test_following(self):
        sigma = self.target
        evLanczos, uvLanczos,status = inexactLanczosDiagonalization(self.mat,self.guess,sigma,self.L,
                self.maxit,self.eConv,pick=self.pick,writeOut=self.writeOut)

        with self.subTest("eigenvalue"):
            evCalc = evLanczos[0]
            relError = abs(evCalc-self.energyRef)/(max(abs(self.energyRef), 1e-14))
            self.assertTrue((relError <= 1e-4),f'{evLanczos=}; reference: {self.energyRef=} ; {self.evEigh:}; \n Not accurate up to 1e-4')
        with self.subTest("eigenvector"):
            ovlp = self.ovlpRef.vdot( uvLanczos[0] )
            np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg = f"{ovlp=} but it should be +-1")

if __name__ == "__main__":
    unittest.main()
