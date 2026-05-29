import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from feast import feastDiagonalization 
from ttnsVector import TTNSVector
from util_funcs import find_nearest
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import orthogonalize 


#######################################################
MAX_D = 3
# 5e-9 ok
EPS = 5e-9
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
#np.random.seed(898989)
#tns.setRandom(dtype=complex)
#tns.toPdf()
#tns.label = "CH3CN using CSC PES"

if EPS is not None:
    bondDimensionAdaptions = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
else:
    bondDimensionAdaptions = None

# ---------- USER INPUT -----------------------
Emin = 720  # Lower limit of excitation energy for target interval
Emax = 730   # Upper limit of excitation energy for target interval
maxit = 3 
n_quad = 6
eps = 1e-6 
quad = "legendre"
zpve = 9837.4069

# ---------- USER INPUT -----------------------

bondAdaptFitting = [TruncationEps(EPS, maxD=20, offset=2, truncateViaDiscardedSum=False)]
optionsLinear = {"nSweep":1000, "iterativeLinearSystemOptions":
        IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-4,maxIter=1000),
        "convTol":1e-4,"bondDimensionAdaptions":bondDimensionAdaptions}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFitting}
options = {"linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}

N_SUBSPACE = 4
# Random orthogonal tress
setTrees = []
for i in range(N_SUBSPACE):
    tns = parseTree(treeString, basisDict, returnType="TTNS")
    np.random.seed(20+i);tns.setRandom(dtype=complex)
    setTrees.append(tns)
setTrees= orthogonalize(setTrees)
        
# Make a TTNSVector list from above orthogonal trees
guess = []
for i in range(N_SUBSPACE):
    guess.append(TTNSVector(setTrees[i],options))

ev_min = util.unit2au((Emin+zpve),"cm-1")  # lower limit of eigenvalue in a.u.
ev_max = util.unit2au((Emax+zpve),"cm-1")  # upper limit of eigenvalue in a.u.

ev, tnsList = feastDiagonalization(Hop,guess,n_quad,quad,ev_min,ev_max,eps,maxit,
        eShift=zpve,convertUnit="cm-1")[0:2]
print("Eigenvalues",util.au2unit(ev,"cm-1")-zpve)
# -----------------   EOF  -----------------------
