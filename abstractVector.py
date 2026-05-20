from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg as la

# Abstract methods are listed here and implemented elsewhere by concrete vector classes.

LINDEP_DEFAULT_VALUE = 1e-14

# Use abstractmethod whenever a concrete vector class must define the behavior.
class AbstractVector(ABC):
    
    @property
    @abstractmethod
    def hasExactAddition(self):
        """
        Simplification of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        but is not exactly identical to 2c for TNSs
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError
   
    @property
    @abstractmethod
    def maxD(self) -> int:
        """Return the maximum virtual bond dimension of a vector (only used for TTNSs)."""
        raise NotImplementedError
   
    @abstractmethod
    def __mul__(self,other):
        raise NotImplementedError
    
    @abstractmethod
    def __rmul__(self,other):
        raise NotImplementedError
    
    @abstractmethod
    def __truediv__(self,other):
        raise NotImplementedError

    @abstractmethod
    def __imul__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __itruediv__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError
    
    @abstractmethod
    def normalize(self):
        """Normalize in place."""
        raise NotImplementedError
        
    @abstractmethod
    def norm(self) -> float:  
        raise NotImplementedError
    
    @abstractmethod
    def real(self):
        raise NotImplementedError

    @abstractmethod
    def conjugate(self):
        raise NotImplementedError

    @abstractmethod
    def vdot(self,other,conjugate=True):
        raise NotImplementedError
     
    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename, additionalInformation:dict=None):
        """Save vector and optional calculation data."""
        raise NotImplementedError
    
    @abstractmethod
    def applyOp(self,other):
        """Apply rmatmul as other@self.array."""
        raise NotImplementedError

    @abstractmethod
    def compress(self):
        """Compress a vector if it is compressible.
        Note: This may return either a copy or a reference to `self`."""
        raise NotImplementedError

    @staticmethod
    def linearCombination(other,coeff):
        """
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        """
        raise NotImplementedError

    @staticmethod
    def orthogonalize(xs,lindep = LINDEP_DEFAULT_VALUE):
        raise NotImplementedError

    @staticmethod
    def orthogonalize_against_set(x,xs,lindep=LINDEP_DEFAULT_VALUE):
        """
        Orthogonalize a vector against the previously obtained set of
        orthogonalized vectors
        x (In): vector to be orthogonalized
        xs (In): set of orthogonalized vectors
        lindep (optional): parameter used to detect linear dependency
        If no linearly independent vector is found with respect to xs, return None.
        """
        raise NotImplementedError
    
    @staticmethod
    def solve(H, b, sigma, x0=None, opType="her",reverseGF=False):
        """Solve the linear equation (sigma*I-H) x = b.

        :param opType: Operator type:
            "gen" for generic operator, "sym" for (complex) symmetric, "her" for Hermitian,
            "pos" for positive definite

         param reverseGF:
             False for Green's function (sigma-H)
             True for reverse Green's function (H-sigma)
        """
        raise NotImplementedError

    @staticmethod
    def matrixRepresentation(operator,vectors):
        """Calculate and return the matrix representation in the "vectors" space of a Hermitian operator."""
        raise NotImplementedError
    
    @staticmethod
    def overlapMatrix(vectors):
        """Calculate the overlap matrix of vectors."""
        raise NotImplementedError
    
    @staticmethod
    def extendMatrixRepresentation(operator,vectors,opMat):
        """Extend the existing operator matrix representation (opMat)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended matrix representation (opMat)"""

        raise NotImplementedError
    
    @staticmethod
    def extendOverlapMatrix(vectors,overlap):
        """Extend the existing overlap matrix (overlap)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended overlap matrix (overlap)"""
        
        raise NotImplementedError
