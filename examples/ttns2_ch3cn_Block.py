import util
from magic import *
import mctdh_stuff
import basis
import copy
import sys
import time
from mpiWrapper import MPI
#MPI.activateMPI()
import operatornD
from ttns2.driver import eigenStateComputations
from ttns2.diagonalization import IterativeDiagonalizationOptions
from ttns2.parseInput import parseTree
from ttns2.contraction import TruncationEps
from ttns2.misc import mpsToTTNS, getVerbosePrinter
from inexact_Lanczos import inexactLanczosDiagonalization
from ttnsVector import TTNSVector
from ttns2.diagonalization import IterativeLinearSystemOptions
from ttns2.driver import computeResidual
from ttns2.state import loadTTNSFromHdf5


#######################################################
MAX_D = 10 
N_BLOCK = 2
target = 360 # 2057 
zpve = 9837.4069  

L = 10 
maxit = 20
eConv = 1e-6
EPS = 5e-9
bondAdaptLinear = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptOrtho = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)] * 1
bondAdaptFit = [TruncationEps(EPS, maxD=MAX_D, offset=2, truncateViaDiscardedSum=False)]
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
tns.toPdf("guess_forDMRG.pdf")
tns.label = "CH3CN using CSC PES"

print("------------------ DMRG for init guess ----------")
tnsList, energies = eigenStateComputations(tns, Hop,
                                           nStates=N_BLOCK,
                                           nSweep=12,
                                           returnIfBelowOptVal = util.unit2au(1.1*(target+zpve),"cm-1"),
                                           allowRestart=False,
                                           projectionShift=util.unit2au(8499,"cm-1"))
print("-------------------------------")
assert(len(tnsList)==N_BLOCK)
###############
optionsOrtho = {"nSweep":40, "convTol":1e-2, "bondDimensionAdaptions":bondAdaptOrtho}
optsCheck = IterativeLinearSystemOptions(solver="gcrotmk",tol=1e-2,maxIter=10)
optionsLinear = {"nSweep":5, "iterativeLinearSystemOptions":optsCheck,"convTol":5e-2,
        "bondDimensionAdaptions":bondAdaptLinear,"shiftAndInvertMode":True, 
        "optValUnit":"cm-1","optShift":util.unit2au(zpve,"cm-1")}
optionsFitting = {"nSweep":1000, "convTol":1e-9,"bondDimensionAdaptions":bondAdaptFit}
options = {"orthogonalizationArgs":optionsOrtho, "linearSystemArgs":optionsLinear, "stateFittingArgs":optionsFitting}
guess = [TTNSVector(t,options) for t in tnsList]

sigma = util.unit2au((target+zpve),unit="cm-1")
ev, tnsList, status = inexactLanczosDiagonalization(Hop,guess,sigma,L,maxit,
        eConv,checkFitTol=1e-3,eShift=zpve,convertUnit="cm-1")
print(status)

# ----------------- Saving Lanczos tress ---------
directory = "finalLanczosTNSs/"
if not os.path.exists(directory):
    os.makedirs(directory)

nvectors = len(tnsList)
for ivec in range(nvectors):
    filename = directory + "lanczosSolution"+str(ivec)+".h5"
    Info = {"energy": ev[ivec],"converged":status["isConverged"],
        "L":L,"target":target,"MAX_D":MAX_D,"N_BLOCK":N_BLOCK} 
    tnsList[ivec].ttns.saveToHDF5(filename,additionalInformation=Info)

# ----------------- Residuals --------------------
outfile = open("iterations_lanczos.out","a") 
outfile.write(r"Norm of residuals (H\Psi - E\Psi)"+"\n\n")
formatStyle = "{:20} :: {:<20}"
line = formatStyle.format("Eigenvalue","Norm in cm-1")
outfile.write(line+"\n")

ntotal = len(ev)
residual_norm = np.empty((ntotal),dtype=float)
for i in range(ntotal):
    psi = tnsList[i].normalize()
    Enew = TTNSVector.matrixRepresentation(Hop,[psi])[0,0]
    residual = computeResidual(psi.ttns,Hop,Enew,
            nSweep=15,convTol=1e-2)
    residual_norm[i] = util.au2unit(residual.norm(),"cm-1")

    Enew = util.au2unit(Enew, "cm-1")-zpve
    line = formatStyle.format(Enew,residual_norm[i])
    outfile.write(line+"\n")

outfile.write("\n\n");outfile.close()
# -----------------   EOF  -----------------------
