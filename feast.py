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

SUBSPACE_CONSTRUCTION_OPTIONS = {
        1:"fitted_sums",
        "fitted_sums":"fitted_sums",
        2:"double_sums",
        "double_sums":"double_sums",
        3:"expanded_space",
        "expanded_space":"expanded_space",
        }

def _canonicalizeSubspaceConstruction(subspaceConstruction):
    try:
        return SUBSPACE_CONSTRUCTION_OPTIONS[subspaceConstruction]
    except KeyError as exc:
        raise ValueError(
                "subspaceConstruction must be 1, 2, 3, "
                "'fitted_sums', 'double_sums', or 'expanded_space'."
                ) from exc

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

def _flattenQuadratureVectors(Qquad):
    """Return a flat list and index map for quadrature-resolved vectors."""
    flat = []
    index = {}
    for iGuess, vectors in enumerate(Qquad):
        for iQuad, vector in enumerate(vectors):
            index[(iGuess,iQuad)] = len(flat)
            flat.append(vector)
    return flat,index

def _sumQuadratureVectors(Qquad):
    """Sum quadrature-resolved vectors for each initial guess."""
    Q = []
    for vectors in Qquad:
        typeClass = vectors[0].__class__
        coeffs = np.ones(len(vectors))
        Q.append(typeClass.linearCombination(vectors,coeffs))
    return Q

def _buildDoubleSumMatrices(A,Qquad,typeClass):
    """Build FEAST matrices by explicit sums over quadrature vectors."""
    flat,index = _flattenQuadratureVectors(Qquad)
    Hflat = typeClass.matrixRepresentation(A,flat)
    Sflat = typeClass.overlapMatrix(flat)
    nGuess = len(Qquad)
    Hmat = np.zeros((nGuess,nGuess),dtype=Hflat.dtype)
    Smat = np.zeros((nGuess,nGuess),dtype=Sflat.dtype)
    for iGuess in range(nGuess):
        for jGuess in range(nGuess):
            for iQuad in range(len(Qquad[iGuess])):
                iFlat = index[(iGuess,iQuad)]
                for jQuad in range(len(Qquad[jGuess])):
                    jFlat = index[(jGuess,jQuad)]
                    Hmat[iGuess,jGuess] += Hflat[iFlat,jFlat]
                    Smat[iGuess,jGuess] += Sflat[iFlat,jFlat]
    return Hmat,Smat,flat

def _backTransformDoubleSum(Qquad,uSH):
    """Back-transform from explicit double-sum FEAST matrices."""
    flat,_ = _flattenQuadratureVectors(Qquad)
    coeffs = np.repeat(uSH,len(Qquad[0]),axis=0)
    return basisTransformation(flat,coeffs)

def _matchedEigenvalueResidual(ev,reference,eMin,eMax):
    """Compare sorted interval eigenvalues from two FEAST iterations."""
    evInside = select_within_range(ev,eMin,eMax)[0]
    refInside = select_within_range(reference,eMin,eMax)[0]
    if len(evInside) == 0 or len(refInside) == 0:
        evInside = np.sort(ev)
        refInside = np.sort(reference)

    evInside = np.sort(evInside)
    refInside = np.sort(refInside)
    if len(evInside) > len(refInside):
        indices = np.argmin(np.abs(evInside[:,None] - refInside[None,:]),axis=0)
        evInside = evInside[indices]
    elif len(refInside) > len(evInside):
        indices = np.argmin(np.abs(refInside[:,None] - evInside[None,:]),axis=0)
        refInside = refInside[indices]

    return eigenvalueResidual(evInside,refInside)

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

# ***************************************************
# Part 1: main FEAST function for contour integral
# ------------------------------
def feastDiagonalization(A, Y: list[AbstractVector],
                         n_quad, quad, eMin, eMax, eConv, maxit, contourEllipseFactor=1.0,
                         writeOut=True, eShift=0.0, 
                         convertUnit="au", outFileName=None, summaryFileName=None,
                         subspaceConstruction="fitted_sums"):
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
    subspaceConstruction (optional) => FEAST subspace construction.
        Default: "fitted_sums".
        1 or "fitted_sums": sum quadrature vectors with linearCombination/fitting.
        2 or "double_sums": build H and S by explicit double sums over
            quadrature vectors, then fit only the next guesses.
        3 or "expanded_space": diagonalize in the full space spanned by
            number-of-guesses times number-of-grid vectors. This can return
            eigenvalues outside the target interval.

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
    subspaceConstruction = _canonicalizeSubspaceConstruction(subspaceConstruction)
    

    # Numerical quadrature points.
    gk, wk = quadraturePointsWeights(n_quad, quad, positiveHalf=True)
    pi = np.pi
    
    status = _getStatus(None,Y)
    status["subspaceConstruction"] = subspaceConstruction
    printObj = FeastPrintUtils(Y, n_quad, quad, eMin, eMax, eConv, maxit, 
            writeOut, eShift, convertUnit, status, subspaceConstruction,
            outFileName, summaryFileName)

    printObj.fileHeader()
    
    for it in range(maxit):
        status["outerIter"] = it
        Qquad = [[] for _ in range(N_SUBSPACE)]
        for k in range(len(gk)):
            status["quadrature"] = k

            # Polizzi (13,14); Baiardi uses slightly different equation
            theta = -(pi*0.5)*(gk[k]-1) # Polizzi (13)
            # z =(eMin + eMax) * 0.5 + eRadius  * exp(2pi i theta)
            # Allow an ellipse instead of a circle on the imaginary axis.
            z = (eMin + eMax) * 0.5 + eRadius * (math.cos(theta) + contourEllipseFactor * 1.0j * math.sin(theta) )
            
            for im0 in range(N_SUBSPACE):
                Qquad_k = calculateQuadrature(A,Y[im0],z,eRadius,theta,wk[k],contourEllipseFactor)
                Qquad[im0].append(Qquad_k)
        
        if subspaceConstruction == "fitted_sums":
            Q = _sumQuadratureVectors(Qquad)
            basisVectors = Q
            Smat = typeClass.overlapMatrix(Q)
            Hmat = typeClass.matrixRepresentation(A,Q)
        elif subspaceConstruction == "double_sums":
            Hmat,Smat,basisVectors = _buildDoubleSumMatrices(A,Qquad,typeClass)
        elif subspaceConstruction == "expanded_space":
            basisVectors,_ = _flattenQuadratureVectors(Qquad)
            Smat = typeClass.overlapMatrix(basisVectors)
            Hmat = typeClass.matrixRepresentation(A,basisVectors)
        else:
            raise RuntimeError(f"Unexpected {subspaceConstruction=}")
        
        if printObj is not None:
            printObj.writeFile("iteration",status)
            printObj.writeFile("overlap",Smat)
        
        status, uS = lowdinOrthoMatrix(Smat,status)
        ev, uv = diagonalizeHamiltonian(uS,Hmat,printObj)
        
        uSH = uS@uv
        del uv
        if subspaceConstruction == "double_sums":
            Y = _backTransformDoubleSum(Qquad,uSH)
        else:
            Y = basisTransformation(basisVectors,uSH)
        del Qquad
        del basisVectors

        if it != 0:
            residual = _matchedEigenvalueResidual(ev,ref_ev,eMin,eMax)
            status["runTime"] = time.time() - status["startTime"]
            status["residual"] = residual
            printObj.writeFile("summary",ev,residual,status)
            
            if residual < eConv:
                break

        if len(Y) < N_SUBSPACE:
            warnings.warn(f"Alert! Got {N_SUBSPACE-len(Y)} \
                dependent vectors")
        elif len(Y) > N_SUBSPACE and subspaceConstruction == "expanded_space":
            warnings.warn(
                    f"expanded_space increased the FEAST guess count from "
                    f"{N_SUBSPACE} to {len(Y)}."
                    )

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
