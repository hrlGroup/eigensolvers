import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
from mpiWrapper import MPI
# MPI is not yet working for Lanczos
#MPI.activateMPI()
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from inexact_Lanczos import inexactLanczosDiagonalization
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from ttns2.diagonalization import IterativeLinearSystemOptions


timeStarting = time.time()
#######################################################
MAX_D = 10 
N_BLOCK = 1
target = 360 # 2057 
zpve = 9837.4069  

L = 10 
maxit = 20
eConv = 1e-4
EPS = 5e-9
N_STATES = 8
bondAdaptLinear = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptOrtho = [TruncationEps(EPS, maxD=L*MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptFit = [TruncationEps(EPS, maxD=L*MAX_D, offset=2, truncateViaDiscardedSum=False)]
#######################################################
_print = getVerbosePrinter(True)
_print("# EPS=",EPS)

fOp = 'ch3cn.op'  # this one is used for HRL's 2019 jcp work
Hop = mctdh_stuff.translateOperatorFile(fOp, verbose=False)
_print("# Hop: nSum=",Hop.nSum)

N = 42
DVRopts = [
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N , HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
    basis.Hermite.getOptions(N=N, HOx0=0, HOw=1, HOm=1),
]
treeString = """
0> 3 3 3
    1> 3 3
        2> [x1]
        2> [x5 x6]
    1> 3 3
        2> [x7 x8]
        2> [x9 x10]
    1> 3 3
        2> 3 3
            3> [x3]
            3> 3 3
                4> [x2]
                4> [x4]
        2> 3
            3> [x11 x12]
    """
bases = [basis.basisFactory(o) for o in DVRopts]
nBas = [b.N for b in bases]
Hop.storeMatrices(bases)
Hop = operatornD.contractSoPOperatorSimpleUsage(Hop)
operatornD.absorbCoeff(Hop)
_print("# Hop contracted nSum",Hop.nSum)
Hop.obtainMultiplyOp(bases)
basisDict = {l:b for l,b in zip(Hop.DoFlabel, bases)}
tns = parseTree(treeString, basisDict, returnType="TTNS")
np.random.seed(898989)
tns.setRandom()
tns.toPdf()
tns.label = "CH3CN using Carrington PES"

if EPS is not None:
    #bondDimensionAdaptions = [TruncationEps(min(EPS*1e3,1e-3), maxD=20, offset=1, truncateViaDiscardedSum=False)] * 4
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=40, offset=2, truncateViaDiscardedSum=False)] * 4)
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e2,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    #bondDimensionAdaptions.extend([TruncationEps(min(EPS*1e1,1e-3), maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 2)
    #bondDimensionAdaptions.extend([TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1)
    # TODO for larger system than ch3cn, try to start with lower bond dimension
    bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
else:
    bondDimensionAdaptions = None
print("------------------ DMRG for init guess ----------")
PROJECTION_SHIFT = util.unit2au(8499,"cm-1")
tnsList, energies = eigenStateComputations(tns, Hop,
                                           nStates=N_BLOCK,
                                           nSweep=12,
                                           returnIfBelowOptVal = util.unit2au(1.1*(target+zpve),"cm-1"),
                                           allowRestart=False,
                                           projectionShift=PROJECTION_SHIFT)
print("-------------------------------")
assert len(tnsList) == N_BLOCK
###############
print("state follow: sigma")
print("bondAdaptLinear",bondAdaptLinear)
print("bondAdaptFit",bondAdaptFit)

optionsOrtho = {"nSweep":40, "convTol":1e-2, "bondDimensionAdaptions":bondDimensionAdaptions}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-2,maxIter=10)
optionsLinear = {"nSweep":10, "iterativeLinearSystemOptions":optsCheck,"convTol":1e-2,"bondDimensionAdaptions":bondDimensionAdaptions, "shiftAndInvertMode":True, "optValUnit":"cm-1","optShift":util.unit2au(zpve,"cm-1")}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondDimensionAdaptions}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
guess = [TTNSVector(t,options) for t in tnsList]

sigma = util.unit2au((target+zpve),unit="cm-1")
ev, tnsList, status = inexactLanczosDiagonalization(Hop,guess,sigma,L,maxit,eConv,eShift=zpve,convertUnit="cm-1",
                                                    outFileName="iterations_1.out",
                                                    summaryFileName="summary_1.out")
print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
tnsList[0].ttns.setDefault()
convergedStates = [ tnsList[0].ttns, ]
convergedStates[0].label = "CH3CN using Carrington PES; first converged state"
convergedEnergies = [ev[0], ]
operator = {
    "RenormalizedSoPOperator": [Hop,],
    "RenormalizedStateProjector": [convergedStates, np.array(convergedEnergies) + PROJECTION_SHIFT ]
}
print("-"*10)
print("# SECOND RUN")
print("-"*10)
guess = tnsList[1]
guess.ttns.label = "CH3CN using Carrington PES"
guess.ttns.setDefault()
# Need to update options
options["linearSystemArgs"]["auxList"] = convergedStates
guess.options = options
# NOTE: Currently RenormalizedStateProjector only works for bra==ket,
#       so it will not work for getting the matrix representation in the Krylov space.
#       Therefore, only use it for solving the linear system.
ev, tnsList, status = inexactLanczosDiagonalization(Hop,guess,sigma,L,maxit,eConv,
                                                    Hsolve=operator,
                                                    eShift=zpve,convertUnit="cm-1",
                                                    outFileName="iterations_2.out",
                                                    summaryFileName="summary_2.out")
print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
