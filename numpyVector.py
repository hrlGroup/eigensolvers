import numpy as np
import scipy
import inspect
from scipy import linalg as la
from abstractVector import AbstractVector, LINDEP_DEFAULT_VALUE
from scipy.sparse.linalg import LinearOperator
import warnings
from scipy.sparse.linalg import spsolve
import math
from scipy.sparse import csc_matrix
from pathlib import Path


def _callIterativeSolver(solver, linOp, rhs, x0, tol, atol, maxiter):
    # "tol" was removed in newer SciPy releases.
    kwargs = {"x0": x0, "maxiter": maxiter}
    parameters = inspect.signature(solver).parameters
    if "rtol" in parameters:
        kwargs["rtol"] = tol
    else:
        kwargs["tol"] = tol
    if "atol" in parameters:
        kwargs["atol"] = atol
    return solver(linOp, rhs, **kwargs)


####################################################################
# Create a NumpyVector class with the required elementary operations.
####################################################################

# file1: abstractVector.py defines the abstract method list.
# file2: numpyVector.py implements those methods for ndarray.
# main.py: uses these classes for the main calculation, such as inexact_Lanczos.py.

# Implement the methods defined by the abstract class.
# -------------------------------------------------------------
class NumpyVector(AbstractVector):

    def __init__(self,array,options=dict()):
        self.array = array
        self.size = array.size
        self.shape = array.shape
        self.options = dict()
         
        opt = options.get("linearSystemArgs",dict())
        opt["linearSolver"] = opt.get("linearSolver", "minres")
        opt["linearIter"] = opt.get("linearIter", 1000)
        opt["linear_tol"] = opt.get("linear_tol", 1e-4)
        opt["linear_atol"] = opt.get("linear_atol", 1e-4)
        self.options["linearSystemArgs"] = opt
    
    @property
    def hasExactAddition(self):
        """
        Simplification of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        but is not exactly identical to 2c for TTNS.
        """
        return True
        
    @property
    def dtype(self):
        return self.array.dtype
     
    @property
    def maxD(self) -> int:
        # Zero means no virtual bonds; this is an isolated vector.
        return 0

    def __mul__(self,other):
        return NumpyVector(self.array*other,self.options)
    
    def __rmul__(self,other):
        return NumpyVector(self.array*other,self.options)

    def __truediv__(self,other):
        return NumpyVector(self.array/other,self.options)
    
    def __imul__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError


    def __len__(self) -> int:
        return len(self.array)
    
    def normalize(self):
        self.array /= la.norm(self.array)
        return self

    def norm(self) -> float:
        return la.norm(self.array)

    def real(self):
        return NumpyVector(np.real(self.array),self.options)

    def conjugate(self):
        return NumpyVector(self.array.conj(),self.options)

    def vdot(self,other,conjugate:bool=True):
        if conjugate:
            return np.vdot(self.array,other.array)
        else:
            return np.dot(self.array.ravel(),other.array.ravel())
    
    def copy(self):
        return NumpyVector(self.array.copy(), self.options)

    def save(self, filename, additionalInformation:dict=None):
        filename = Path(filename)
        if filename.suffix != ".npz":
            filename = filename.with_suffix(".npz")
        saveDict = {"vector": self.array}
        if additionalInformation is not None:
            saveDict.update(additionalInformation)
        np.savez(str(filename), **saveDict)

    def applyOp(self,other):
        """Apply rmatmul as other@self.array."""
        return NumpyVector(other@self.array,self.options)

    def compress(self):
        return self

    def linearCombination(vectors,coeffs):
        """
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: vectors == list of vectors
             coeffs == list of coefficients, [c1,c2,...,cn]
        """
        assert len(vectors) == len(coeffs)
        dtype = vectors[0].dtype
        combArray = np.zeros(len(vectors[0]),dtype=dtype)
        for n in range(len(vectors)):
            combArray += coeffs[n]*vectors[n].array
        return NumpyVector(combArray,vectors[0].options)

    def orthogonalize_against_set(x, qs, lindep=LINDEP_DEFAULT_VALUE):
        """
        Orthogonalize a vector against the previously obtained set of
        orthogonalized vectors
        x (In): vector to be orthogonalized
        qs (In): set of orthogonalized vectors
        lindep (optional): parameter used to detect linear dependency
                          Default value is LINDEP_DEFAULT_VALUE
                          See module abstractVector.py
        If no linearly independent vector is found with respect to qs, return None.
        """
        nv = len(qs)
        for i in range(nv):
            qsi = qs[i]
            term1 = x.vdot(qsi,conjugate=False)
            term2 = qsi.vdot(qsi,conjugate=False)
            proj = qsi*(term1/term2)
            x = NumpyVector.linearCombination([x,proj],[1.0,-1.0])
            #x = x - proj
        innerprod = x.vdot(x,conjugate=False)
        if innerprod > lindep:
            x = x/np.sqrt(innerprod) # normalize
        else:
            x = None
        return x

    @staticmethod
    def solve(H, b, sigma, x0=None, opType="her", reverseGF=False):
        n = H.shape[0]
        dtype = np.result_type(sigma, H.dtype, b.dtype)
        if not reverseGF:
            linOp = LinearOperator((n,n),matvec = lambda x, sigma=sigma, H=H:(sigma*x-H@x),dtype=dtype)
        elif reverseGF:
            linOp = LinearOperator((n,n),matvec = lambda x, sigma=sigma, H=H:(H@x-sigma*x),dtype=dtype)
        
        options = b.options["linearSystemArgs"]
        tol = options["linear_tol"]
        atol = options["linear_atol"]
        maxiter = options["linearIter"]
        if options["linearSolver"] == "gcrotmk":
            wk,conv = _callIterativeSolver(scipy.sparse.linalg.gcrotmk, linOp, b.array, x0, tol, atol, maxiter)
        elif options["linearSolver"] == "minres":
            wk,conv = _callIterativeSolver(scipy.sparse.linalg.minres, linOp, b.array, x0, tol, None, maxiter)
        elif options["linearSolver"] == "pardiso": # only for comparison with Fortran
            if not reverseGF:
                A1 = csc_matrix(sigma*np.eye(n)-H)
            else:
                A1 = csc_matrix(H-sigma * np.eye(n))
            b1 = csc_matrix(np.reshape(b.array,(n,1)))
            wk = spsolve(A1,b1)
            conv = 0 # Direct solve; treat it as converged.
        else:
            raise Exception("Got linear solver other than gcrotmk, minres and pardiso!")

        if conv != 0:
            warnings.simplefilter('error', UserWarning)
            warnings.warn("Warning: iterative solver did not converge.")
        return NumpyVector(wk,b.options)

    def matrixRepresentation(operator,vectors):
        """Calculate and return the matrix representation in the "vectors" space."""
        m = len(vectors)
        dtype = vectors[0].dtype
        qtAq = np.zeros((m,m),dtype=dtype)
        for j in range(m):
            ket = vectors[j].applyOp(operator)
            for i in range(j,m):
                qtAq[i,j] = vectors[i].vdot(ket)
                qtAq[j,i] = qtAq[i,j].conj()
        return qtAq

    def overlapMatrix(vectors):
        """Calculate the overlap matrix of vectors."""

        m = len(vectors)
        dtype = vectors[0].dtype
        Smat = np.zeros((m,m),dtype=dtype)
        
        for i in range(m):
            for j in range(i,m):
                Smat[i,j] = vectors[i].vdot(vectors[j],True)
                Smat[j,i] = Smat[i,j].conj()
        return Smat
    
    def extendMatrixRepresentation(operator,vectors,opMat):
        """Extend the existing operator matrix representation (opMat)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended matrix representation (opMat)"""

        m = len(vectors)
        dtype = vectors[0].dtype

        elems = np.empty((1,m),dtype=dtype)
        ket = vectors[-1].applyOp(operator)
        for i in range(m):
            elems[0,i] = vectors[i].vdot(ket)
        opMat = np.append(opMat,elems[:,:-1].conj(),axis=0)
        opMat = np.append(opMat,elems.T,axis=1)
        return opMat

    def extendOverlapMatrix(vectors,overlap):
        """Extend the existing overlap matrix (overlap)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended overlap matrix (overlap)"""
        
        m = len(vectors)
        dtype = vectors[0].dtype

        elems = np.empty((1,m),dtype=dtype)
        for i in range(m):
            elems[0,i] = vectors[i].vdot(vectors[-1],True)
        overlap = np.append(overlap,elems[:,:-1].conj(),axis=0)
        overlap = np.append(overlap,elems.T,axis=1)
        return overlap
