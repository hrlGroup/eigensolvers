import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy import special
from util_funcs import select_within_range, quadraturePointsWeights, eigenvalueResidual
from util_funcs import basisTransformation
from util_funcs import lowdinOrthoMatrix, diagonalizeHamiltonian 
from numpyVector import NumpyVector
from abstractVector import AbstractVector
import warnings
import time
import math
from magic import ipsh
from printUtils import FeastPrintUtils

def _getStatus(status,guess):
    """Build a dictionary for storing the computation status.
    In: guess: guess vector used to access the exact-addition property
    
    flagAddition: Boolean, True if the linear combination is accurate
    isConverged: status of eigenvalue residual convergence
    phase: stage of phase calculations
    startTime: starting time
    runTime: run time in seconds

    Out: StatusUp: initialized and updated dictionary
    """

    statusUp ={"flagAddition":guess[0].hasExactAddition,
               "outerIter":0, "quadrature":0,
               "isConverged":False,
               "phase":1,
               "residual":None,
               "startTime":time.time(), "runTime":0.0}
    
    if status is not None:
        givenkeys = status.keys()
    
        for item in givenkeys:       # overwrite defaults
            if item in status:
                statusUp[item] = status[item]
    
    return statusUp

def calculateQuadrature(Amat,guess_b,z,radius,angle,weight,contourEllipseFactor):
    """Calculate k-th quadrature Qquad_k assuming `Amat` is Hermitian.
    
    For Hermitian matrix:
    Qquad_k=-0.25*w_k*r{exp(i*angle_k)G(z)Y+exp(-i*theta)G^dagger(z)Y}
    
    For a real symmetric matrix:
    Qquad_k=-0.50*w_k*Real{r*exp(i*angle_k)G(z)Y}
    
    G(z)Y == Qe => From linear solver: (z*I-A)Qe = Y

    In: Amat => Matrix A for the problem Ax = ex
                Either as ndarray, linear operator or SOP
        guess_b => Guess for b for solving Qe as in linear system Ax = b
        z => k-th contour point
        radius => radius of the contour
        angle => k-th contour angle
        weight => k-th quadrature distribution weight
        contourEllipseFactor => contour shape factor (see below)
    
    Out: Qquad_k => k-th quadrature vector
    Note: exp(+/-i*theta) is expanded as
    contourEllipseFactor*cos(theta)+/-isin(theta)
    This changes the contour shape.
    e.g., contourEllipseFactor = 1.0, circular contour
          contourEllipseFactor = 0.3, ellipse contour
    
    This contourEllipseFactor is implemented in Polizzi's code.
    It is necessary for testing Fortran data.
    """

    b = guess_b # copy the guess so the original is not altered
    typeClass = b.__class__
    if abs(z.imag) < 1e-15:
        # Some contours are on the real axis only
        opType = "her"
        z = z.real
    else:
        # Assuming Amat is Hermitian
        # TODO `sym` seems to have some numerical stability problems in scipy.solve
        #       TTNS unit tests don't converge.
        #opType = "sym"
        opType = "gen"

    if b.hasExactAddition:
        Qe = typeClass.solve(Amat,b,z, opType=opType)  # complex128
        mult = -0.50*weight*radius*(contourEllipseFactor*math.cos(angle)+math.sin(angle)*1j)
        Qquad_k = typeClass.real(mult*Qe)
    else:
        # Polizzi (12)
        mult = -0.25*weight*radius
        part1 = typeClass.solve(Amat,b,z,opType=opType)
        part2 = typeClass.solve(Amat,b,z.conj(),opType=opType)
        c1 = mult*(contourEllipseFactor*math.cos(angle)+math.sin(angle)*1j)
        c2 = mult*(contourEllipseFactor*math.cos(angle)-math.sin(angle)*1j)
        #print("Fit: calculateQuadrature")
        Qquad_k = typeClass.linearCombination([part1,part2],[c1,c2])

    return Qquad_k

def updateQ(Q,im0,Qquad_k,k):
    """Add the k-th quadrature solution to the existing Q.
    In: Q => Q vectors
        im0 => im0-th vector to be updated
        Qquad_k => k-th quadrature for the im0-th 
                vector to be updated
        k => quadrature point

    Out: Q => updated Q vectors"""

    typeClass = Qquad_k.__class__
    if k == 0:
        Q[im0] = Qquad_k
    else:
        #print("Fit: Update quadrature")
        Q[im0] = typeClass.linearCombination([Q[im0],Qquad_k],[1.0,1.0])
    return Q
       
# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feastDiagonalization(A, Y: list[AbstractVector],
                         n_quad, quad, eMin, eMax, eConv, maxit, contourEllipseFactor=1.0,
                         writeOut=True, eShift=0.0, 
                         convertUnit="au", outFileName=None, summaryFileName=None):
    """FEAST diagonalization of A.

    See Polizzi, PRB, 79, 115112 (2009) 10.1103/PhysRevB.79.115112
    and Baiardi, Kelemen, Reiher, JCTC, 18, 1415 (2021) 10.1021/acs.jctc.1c00984

    Input parameters
    ----------------
    A => matrix, linear operator, or SOP operator
         Note: Must be Hermitian. 
         Otherwise, `calculateQuadrature` needs to be adapted.
    Y => Initial guess of vectors.
    n_quad => number of quadrature points
    quad => quadrature points distribution
            Available options - "legendre", "hermite", "trapezoidal"
            Note: Hermite will lead to points outside the [eMin, eMax] 
                  interval.
    eMin => eigenvalue lower limit
    eMax => eigenvalue upper limit
    eConv => eigenvalue residual convergence tolerance
             Residual is calculated through 
             Sum |E - Eprev| / sum(abs(E)),
             where E (Eprev) is the eigenvalue vector of the current 
             (previous) iteration.
    maxit => maximum feast iterations
    contourEllipseFactor (optional) => contour shape factor
                                       See `calculateQuadrature`
    writeOut (optional) => whether to write output files
    eShift (optional) => shift value for printing. 
                         Assuming `A` is shifted by this value.
    convertUnit (optional) => unit for printing
    outFileName (optional) => output file name
    summaryFileName (optional) => summary file name

    Output parameters
    ----------------
    ev =>  feast eigenvalues
    Y  =>  feast eigenvectors
    status => information dictionary
    """

    typeClass = type(Y[0])
    N_SUBSPACE = len(Y)
    assert eMax > eMin
    eRadius = (eMax - eMin) * 0.5
    

    # Numerical quadrature points.
    gk, wk = quadraturePointsWeights(n_quad, quad, positiveHalf=True)
    pi = np.pi
    
    status = _getStatus(None,Y)
    printObj = FeastPrintUtils(Y, n_quad, quad, eMin, eMax, eConv, maxit, 
            writeOut, eShift, convertUnit, status, 
            outFileName, summaryFileName)

    printObj.fileHeader()
    
    for it in range(maxit):
        status["outerIter"] = it
        # initialize Q
        Q = [np.nan for it in range(N_SUBSPACE)]
        for k in range(len(gk)):
            status["quadrature"] = k

            # Polizzi (13,14); Baiardi uses slightly different equation
            theta = -(pi*0.5)*(gk[k]-1) # Polizzi (13)
            # z =(eMin + eMax) * 0.5 + eRadius  * exp(2pi i theta)
            # Allow an ellipse instead of a circle on the imaginary axis.
            z = (eMin + eMax) * 0.5 + eRadius * (math.cos(theta) + contourEllipseFactor * 1.0j * math.sin(theta) )
            
            for im0 in range(N_SUBSPACE):
                Qquad_k = calculateQuadrature(A,Y[im0],z,eRadius,theta,wk[k],contourEllipseFactor)
                Q = updateQ(Q,im0,Qquad_k,k)
        
        # Solve the eigenvalue problem in the Lowdin orthogonal basis.
        Smat = typeClass.overlapMatrix(Q)
        Hmat = typeClass.matrixRepresentation(A, Q)
        
        if printObj is not None:
            printObj.writeFile("iteration",status)
            printObj.writeFile("overlap",Smat)
        
        status, uS = lowdinOrthoMatrix(Smat,status)
        ev, uv = diagonalizeHamiltonian(uS,Hmat,printObj)
        
        uSH = uS@uv
        del uv
        Y = basisTransformation(Q,uSH)
        del Q

        if it != 0:
            if len(ref_ev) > len(ev):
                # TODO add unit test for this case # not priority
                # Get elements in ref_ev that are closest to ev
                indices = np.argmin(np.abs(ref_ev[:, None] - ev[None, :]) , axis=0)
                ref_ev = ref_ev[indices]
            elif len(ref_ev) < len(ev):
                raise RuntimeError(f"{ref_ev=} but {ev=}. Enlarged space?")
            residual = eigenvalueResidual(ev, ref_ev, [eMin, eMax])
            status["runTime"] = time.time() - status["startTime"]
            status["residual"] = residual
            printObj.writeFile("summary",ev,residual,status)
            
            if residual < eConv:
                break

        if N_SUBSPACE != len(Y): 
            warnings.warn(f"Alert! Got {N_SUBSPACE-len(Y)} \
                dependent vectors")

        N_SUBSPACE = len(Y)
        ref_ev = ev

    printObj.writeFile("results",ev)
    printObj.fileFooter()

    return ev,Y,status


if __name__ == "__main__":
    # ***************************************************
    # Part 1: Call FEAST program with parameter specifications
    # ------------------------------
    n = 100
    ev = np.linspace(1,200,n)
    np.random.seed(10)
    Q = sp.linalg.qr(np.random.rand(n,n))[0]
    A = Q.T @ np.diag(ev) @ Q
    linOp = LinearOperator((n,n), matvec = lambda x, A=A: A@x)

    # Specify FEAST parameters
    ev_min = 160.0
    ev_max = 166.0
    n_quad = 8         # number of quadrature points
    quad  = "legendre" # Choice of quadrature points # available options, legendre, Hermite (, trapezoidal !)
    m0    = 6         # subspace dimension
    eps   = 1e-6      # residual convergence tolerance
    maxit = 4         # maximum FEAST iterations
    options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
    options = {"linearSystemArgs":options}
    
    Y0    = np.random.random((n,m0)) # eigenvector initial guess
    for i in range(m0):
         Y0[:,i] = np.ones(n) * (i+1)
    Y1 = sp.linalg.qr(Y0,mode="economic")[0]


    Y = []
    for i in range(m0):
        Y.append(NumpyVector(Y1[:,i], options))

    contour_ev = select_within_range(ev, ev_min, ev_max)[0]
    print("--- actual eigenvalues",contour_ev,"---\n")
    efeast,ufeast =  feastDiagonalization(linOp,Y,n_quad,quad,ev_min,ev_max,eps,maxit)[0:2]
    print("\n---feast eigenvalues",efeast,"---")
