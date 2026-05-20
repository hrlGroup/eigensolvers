import numpy as np
import scipy as sp
from typing import List, Union
from util_funcs import find_nearest, basisTransformation
from util_funcs import lowdinOrthoMatrix, diagonalizeHamiltonian 
from printUtils import LanczosPrintUtils
import warnings
import time
import util
from abstractVector import AbstractVector
from numpyVector import NumpyVector
from ttnsVector import TTNSVector
from util_funcs import headerBot
from util_funcs import get_pick_function_close_to_sigma
from util_funcs import get_pick_function_maxOvlp
from util_funcs import eigenvalueResidual
import copy
import os

# -----------------------------------------------------
# Dividing in to functions for better readability 
# and convenient testing
def _getStatus(status, guessVector, nBlock):
    """ 
    Initialize and update status dictionary
    
    In: status -> status input dictionary
        guessVector -> guess vector
        nBlock -> Lanczos blocks
    Out: statusUp  -> initialized and updated

    Status contains following information
    (i)     Reference for residual calculations,
            and resdidual
    (ii)    Block info (number of blocks)
    (iii)   Vector flagAddition property
    (iv)    Stage of iteration
    (v)     zeroVector, convergence, lindep and futile restarts
            informations
    (vi)    Run time
    (vii)   Number of phases

    keys: ["ref","residual","nBlock","flagAddition",
    "outerIter","innerIter","cumIter","iBlock",
    "zeroVector", "isConverged","lindep","futileRestarts",
    "startTime","runTime","phase"]


    "ref" is a list of np.arrays -> always contains maximum two items.
    Each item of the list contains nearest n block eigenvalues
    and serves as reference. Purpose of having two 
    items in the reference list: (i) Latest item (or the second element) 
    is used for evaluating convergence residual (see in 
    'checkConvergence' module), and (ii) After evaluating residual
    for convergence check, the "ref" list is updated with current 
    nBlock eigenvalues. At end of Krylov iteration, decision is to 
    be made for terminateRestart for cases with lindep.
    Second element is already the updated nBlock eigenvalues from
    current iteration. Here, the first element serves the purpose
    of reference to check residual of restart.

    zeroVector is True when linear solution has norm less than 0.001*eConv
    """
    
    statusUp = {"ref":[],"residual":np.inf,"nBlock":nBlock,
            "flagAddition":guessVector.hasExactAddition,
            "outerIter":0, "innerIter":0,"cumIter":0,
            "iBlock":0,"zeroVector":False,
            "isConverged":False,"lindep":False,
            "futileRestarts":0,
            "startTime":time.time(), "runTime":0.0,
            "KSmaxD":[],"fitmaxD":None,
            "phase":1}

    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def generateSubspace(Hop, vec:List[AbstractVector],sigma,eConv):
    """ Builds Krylov space by solving linear system
    (Hop-sigma) x = vec
    and subsequent normalization if x is nonzero.
    Nonzero is defined by norm > 0.001*eConv

    In: Hop -> Operator (either as matrix or linearOperator)
        vec -> List of Krylov vectors
        sigma -> Eigenvalue target
        eConv -> Eigenvalue convergence tolerance

    Out: New vector x, nonzero
    """

    typeClass = type(vec)
    out = typeClass.solve(Hop,vec,sigma)
    if typeClass.norm(out) > 0.001*eConv:
        out = typeClass.normalize(out)
        nonzero = True
    else:
        nonzero = False
    return out, nonzero

def _convergence(value, ref):
    ''' Computes convergence quantity (absolute error or 
    relative error, current one is relative error )'''
    
    check_ev = abs(value - ref)/max(abs(value), 1e-14)
    return check_ev


def checkConvergence(ev,eConv,status,printObj=None):
    ''' Checks eigenvalue convergence
    
    In: ev -> eigenvalues, sorted based on `pick`
        status -> params dictionary
        printObj (opional): print object 
    
    Out: status (dict: updated isConverged, ref)
         '''
    
    isConverged = False
    nBlock = status["nBlock"]
    nBlockEigenvalues = np.sort(ev[0:nBlock])   # nBlock states, sort them to avoid root flipping

    # Residual calculation and check for all except cumIter = 1
    if status["cumIter"] > 1:
        reference = status["ref"][-1] 
        residual = eigenvalueResidual(nBlockEigenvalues,reference)
        status["residual"] = residual
        if residual <= eConv:
            isConverged = True

    status["isConverged"] = isConverged
    status["runTime"] = time.time() - status["startTime"]
    if printObj is not None:
        printObj.writeFile("summary",nBlockEigenvalues, status)
    status["ref"].append(nBlockEigenvalues)
    if len(status["ref"]) > 2:status["ref"].pop(0)
    return status
 
def checkFitting(evNew, ev, checkFitTol, status):
    ''' Checks the eigenvalue after fitting
    (at the end of Lanczos iteration)
    In : evNew -> energy after fitting sum of states
         ev -> energy of state before fitting
         checkFit -> checking tolerance of fitted vectors eigenvalues
         status -> Param dictionary
    
    Out: properFit -> (bool: True for accurate linear combination)
    '''
    properFit = True
    
    if status["flagAddition"]:
        properFit = True
    else:
        if _convergence(evNew,ev) > checkFitTol:
           properFit = False
           iBlock = status["iBlock"]
           print(f"Linearcombination inaccurate for {iBlock}: After fit:\
                   {evNew}. Before fit: {ev}")
    return properFit

def terminateRestart(blockEnergies,eConv,status,num=3):
    """ This module looks if Lanczos restarts are fruitful or not
    
    futileRestarts -> Number of ineffective or futile restarts
    If the eigenvalue residual change is greater than max(1e-9,eConv),
    counted as an ineffective or futile restart and adds
    1 to futileRestarts

    In: blockEnergies -> nBlock energies after fitting
        eConv -> eigenvalue convergence
        status -> param dictionary
        num (optional) -> Number of futile restarts
                          Default is 3
    Out: decision (Boolean) -> decision to terminate restart"""
    
    decision = False
    prevBlockEnergies = status["ref"][0]

    if status["lindep"]:
        residual = eigenvalueResidual(blockEnergies,prevBlockEnergies)
        if residual > max(1e-9,eConv):
            status["futileRestarts"] += 1

    if status["futileRestarts"] > num:
        warnings.warn("Lindep and did not have fruitful restarts")
        decision = True

    return decision


def analyzeStatus(status,maxit,L):
    ''' Wrapper of decision parameter for iteration, isConverged'
        in a separate function and conclude to a single 
        bool param continueIteration
        to make main function clean

    In: status -> param dictionary
        maxit -> maximum Lanczos iterations
        L -> Krylov dimension
        
    Out: decision to continue iteration'''

    isConverged = status["isConverged"]
    it = status["outerIter"]
    i = status["innerIter"]
    continueIteration = True
    
    if isConverged:
        continueIteration = False
    
    if it == maxit -1 and i == L-1: 
        if not isConverged: 
            print("Alert: Lanczos iterations is not converged!")
            continueIteration = False

    return continueIteration
# -----------------------------------------------------

# -----------------------------------------------------
#    Inexact Lanczos with AbstractClass interface
#------------------------------------------------------

def inexactLanczosDiagonalization(H,  v0: Union[AbstractVector,List[AbstractVector]],
                                  sigma, L, maxit, eConv, checkFitTol=1e-7,
                                  Hsolve=None,
                                  pick=None, status=None,
                                  writeOut=True, eShift=0.0, convertUnit="au",
                                  outFileName=None, summaryFileName=None,
                                  saveAllVectors=True, saveDir="lanczosVectors"):
    """ Calculate eigenvalues and eigenvectors using the inexact Lanczos method


    ---Doing inexact Lanczos in canonical orthogonal basis.---
    
    Input parameters
    ----------------
             H => diagonalizable input matrix or linearoperator
             v0 => eigenvector guess
                    Can be a list of `AbstractVectors`.
                    Then, block Lanczos is performed (Krylov space on each of the guesses).
                    Note that the guess vectors should be orthogonal.
             sigma => eigenvalue estimate
             L => Krylov space dimension
             maxit => Maximum Lanczos iterations
             eConv => relative eigenvalue convergence tolerance
             checkFitTol (optional) => checking tolerance of fitted vectors
                             eigenvalues
             Hsolve (optional) => As H but only used for the generation of the Lanczos vectors
                    `H` is then used for diagonalizing the Hamiltonian matrix
             writeOut (optional) => writing file instruction
             default : write both iteration_lanczos.out & summary_lanczos.out
             eShift (optional) => shift value for eigenvalues, Hmat elements
             convertUnit (optional) => convert unit for eigenvalues, Hmat elements
             pick (optional) => pick function for eigenstate 
                            Default is get_pick_function_close_to_sigma
             status (optional) => Additional information dictionary
                    (more details see _getStatus doc)
            outFileName (optional): output file name
            summaryFileName (optional): summary file name
            saveAllVectors (optional): save Lanczos vectors to `saveDir` during each outer iteration
            at each cumulative iteration
            saveDir (optional): directory for saving Krylov vectors


    Output parameters
    ----------------
    ev =>  inexact Lanczos eigenvalues
    Y  =>  inexact Lanczos eigenvectors
    status => information dictionary
    """

    if issubclass(type(v0), AbstractVector):
        v0 = [v0]
    else:
        assert isinstance(v0, (list, tuple, np.ndarray)), f"{v0=} {type(v0)=}"
    if Hsolve is None:
        Hsolve = H
    typeClass = type(v0[0])
    nBlock = len(v0)

    Ylist = v0.copy() # Krylov subspace lists.
    Smat = typeClass.overlapMatrix(Ylist)
    if not np.allclose(Smat, np.eye(nBlock), rtol=1e-3, atol=1e-3):
        if nBlock > 1:
            raise RuntimeError(f"Input vectors not orthogonalized: {Smat=}")
        else:
            # gracefully do this. I do not want to do it for nBlock as GS orthogonalization modifies the block space
            Ylist[0].normalize()
            Smat[0,0] = 1
    Hmat = typeClass.matrixRepresentation(H,Ylist)

    status = _getStatus(status,Ylist[0],nBlock)
    if pick is None:
        pick = get_pick_function_close_to_sigma(sigma)
    assert callable(pick)

    printObj = LanczosPrintUtils(Ylist[0],sigma,L,maxit,eConv,checkFitTol,
            writeOut,eShift,convertUnit,pick,status, outFileName, 
            summaryFileName)
    printObj.fileHeader()

    for outerIter in range(maxit):
        status["outerIter"] = outerIter
        status["KSmaxD"] = [Ylist[0].maxD]
        status["fitmaxD"] = None
        for innerIter in range(1,L): # starts with 1 because Y0 is used as first basis vector
            status["innerIter"] = innerIter
            status["cumIter"] += 1
            #
            # Generate subspace
            #
            newVectors = []
            for iBlock in range(1,nBlock+1):
                out, nonzero = generateSubspace(Hsolve, Ylist[-iBlock], sigma, eConv)
                if not nonzero:
                    status["zeroVector"] = True
                    warnings.warn(f"Alert: zero vector: ||inv(H-sigma)vec||={typeClass.norm(out):5.3e}")
                    break
                newVectors.append(out)
            if not nonzero: # break Krylov loop too
                break
            #
            # Orthogonalize and append
            # Note that the new vectors are also orthogonalized against each other.
            # Also extends overlap and Hamiltonian matrices
            #
            lindepProblem = False
            firstNewVector = len(Ylist)
            for iBlock in range(nBlock):
                status["iBlock"] = iBlock
                newOrthVec = typeClass.orthogonalize_against_set(newVectors[iBlock],Ylist)
                if newOrthVec is None:
                    lindepProblem = True
                    status["lindep"] = True
                    if printObj.writeOut:
                        warnings.warn(f"Linear dependency problem in iteration {outerIter} "
                                  f"and microiteration {innerIter} for block state {iBlock},"
                                  f" abort current Lanczos iteration and restart.")
                    # As extension, in principle I can continue with the remaining block iterations.
                    #   But I assume that this here rarely happens
                    break
                Ylist.append(newOrthVec.compress())
                status["KSmaxD"].append(Ylist[-1].maxD)
                # Extend matrices
                Smat = typeClass.extendOverlapMatrix(Ylist, Smat)
                Hmat = typeClass.extendMatrixRepresentation(H, Ylist, Hmat)
            # Overlap info
            if printObj is not None:
                printObj.writeFile("iteration", status)
                printObj.writeFile("overlap", Smat)
                printObj.writeFile("KSmaxD", status)
            if lindepProblem:
                ev = np.array([np.nan] * len(Ylist))
                del uSH, Hmat, Smat # not up to date
                break
            #
            # Diagonalize
            #
            # Transform to orthogonal basis to check once again linear dependencies
            # I could also just solve the generalized eigenvalue problem directly
            # But this way I could avoid the above GS orthogonalization or modify it
            #   to ignore linear dependency problems
            status, uS = lowdinOrthoMatrix(Smat, status)
            assert not status["lindep"] # should have been taken care of above
            ev, uv = diagonalizeHamiltonian(uS,Hmat,printObj)
            uSH = uS@uv
            del uv
            # Reorder uv and ev indices based on `pick`
            idx = pick(uSH,Ylist,ev)
            assert len(idx) == len(ev), f"{len(ev)=} {len(idx)=}"
            ev = ev[idx]
            uSH = uSH[:,idx]
            #
            # Checks
            #
            status = checkConvergence(ev,eConv,status,printObj)
            continueIteration = analyzeStatus(status,maxit,L)
            
            # Save the Lanczos vectors generated in this step.
            if saveAllVectors:
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                for ivector in range(firstNewVector, len(Ylist)):
                    additionalInformation = {"status":status,
                            "eigencoefficients":uSH,"eigenvalues":ev} 
                    filename = f"{saveDir}/vector_{outerIter:03d}_{ivector:03d}"
                    Ylist[ivector].save(filename,
                            additionalInformation=additionalInformation)

            if not continueIteration:
                break
        if lindepProblem:
            break

        if not continueIteration:
            # Finish up and then return
            Ylist = basisTransformation(Ylist,uSH)
            # check orthogonality of S
            Smat = typeClass.overlapMatrix(Ylist)
            if not np.allclose(Smat, np.eye(len(Ylist)), rtol=checkFitTol, atol=checkFitTol):
                warnings.warn(f"Alert:Final eigenvectors are not properly fitted. S=\n{Smat}")
                properFit = False
            else:
                properFit = True
            status["fitmaxD"] = [item.maxD for item in Ylist]
            if printObj is not None:
                printObj.writeFile("fitmaxD", status)
            break
        else:
            # Simple restart of Lanczos iteration using new eigenvectors
            # Could be improved using thick restart
            newGuessList = []
            for iBlock in range(nBlock):
                guess = basisTransformation(Ylist,uSH[:,iBlock])
                guess = typeClass.normalize(guess[0])
                newGuessList.append(guess)
            Ylist = newGuessList
            Smat = typeClass.overlapMatrix(Ylist)
            Hmat = typeClass.matrixRepresentation(H,Ylist)
            # Check accuracy of basis transformation
            if not np.allclose(Smat, np.eye(len(Ylist)), rtol=checkFitTol, atol=checkFitTol):
                warnings.warn(f"Alert:Final eigenvectors are not properly fitted. S=\n{Smat}")
                properFit = False
                break
            else:
                properFit = True
            evNew = sp.linalg.eigvalsh(Hmat, Smat)
            ##################################################
            if terminateRestart(evNew,eConv,status):
                break
            status["fitmaxD"] = [item.maxD for item in Ylist]
            if printObj is not None:
                printObj.writeFile("fitmaxD",status)
    
    printObj.writeFile("results",ev)
    printObj.fileFooter()
    
    return ev, Ylist, status
# -----------------------------------------------------
if __name__ == "__main__":
    n = 100
    ev = np.linspace(1,300,n)
    np.random.seed(10)
    Q = sp.linalg.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q

    target = 30
    maxit = 4
    L = 6 
    eConv = 1e-8

    options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
    optionDict = {"linearSystemArgs":options}
    writeOut = True
    Y0 = NumpyVector(np.random.random((n)),optionDict)
    sigma = target

    t1 = time.time()
    pick =  get_pick_function_close_to_sigma(sigma)
    #pick =  get_pick_function_maxOvlp(Y0)
    lf,xf,status =  inexactLanczosDiagonalization(A, Y0, sigma, L, maxit, eConv, pick=pick, writeOut=writeOut)
    t2 = time.time()

    print("{:50} :: {: <4}".format("Eigenvalue nearest to sigma",round(find_nearest(lf,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Actual eigenvalue nearest to sigma",round(find_nearest(ev,sigma)[1],8)))
    print("{:50} :: {: <4}".format("Time taken (in sec)",round((t2-t1),2)))
