from abc import ABC, abstractmethod
import numpy as np
from scipy import linalg as la

# file1: abstractVector.py holding these abstract functions list
# file2: numpyVector.py :: specifications of the tasks for each functions defined earlier for ndarray
# main.py: utilizing these class for main purpose, such as inexact_Lanczos.py

# Abstract functions are here for initiation/listing
# We name them and pass for later use

LINDEP_DEFAULT_VALUE = 1e-14

# Specify abstractmethod whenever the task should be specified later
class AbstractVector(ABC):
    
    @property
    @abstractmethod
    def hasExactAddition(self):
        """
        Simplication of vector addition with its complex conjugate.
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
        """Returns maximum value of virtual bond dimensions of a vectors (only used for TTNSs)."""
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
        """ Normalizes in-place"""
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
        ''' Apply rmatmul as other@self.array '''
        raise NotImplementedError

    @abstractmethod
    def compress(self):
        """ Compress vector if it is compressible.
        Note: May be a copy or a ref of `self`."""
        raise NotImplementedError

    @staticmethod
    def linearCombination(other,coeff):
        '''
        Returns the linear combination of n vectors [v1, v2, ..., vn]
        combArray = c1*v1 + c2*v2 + cn*vn 
        Useful for addition, subtraction: c1 = 1.0/-1.0, respectively

        In:: other == list of vectors
             coeff == list of coefficients, [c1,c2,...,cn]
        '''
        raise NotImplementedError

    @staticmethod
    def orthogonalize(xs,lindep = LINDEP_DEFAULT_VALUE):
        raise NotImplementedError

    @staticmethod
    def orthogonalize_against_set(x,xs,lindep=LINDEP_DEFAULT_VALUE):
        '''
        Orthogonalizes a vector against the previously obtained set of 
        orthogonalized vectors
        x (In): vector to be orthogonalized 
        xs (In): set of orthogonalized vector
        lindep (optional): Parameter to check linear dependency
        If it does not find linearly independent vector w.r.t. xs; it returns None
        '''
        raise NotImplementedError
    
    @staticmethod
    def solve(H, b, sigma, x0=None, opType="her",reverseGF=False):
        ''' Linear equation (sigma*I-H) x =b solver

        :param opType: Operator type:
            "gen" for generic operator, "sym" for (complex) symmetric, "her" for hermitian,
            "pos" for positive definite

         param reverseGF
             False for Green's function (sigma-H)
             True for reverse Green's function (H-sigma)
        '''
        raise NotImplementedError

    @staticmethod
    def matrixRepresentation(operator,vectors):
        ''' Calculates and returns matrix in the "vectors" space of a *hermitian* operator. '''
        raise NotImplementedError
    
    @staticmethod
    def overlapMatrix(vectors):
        ''' Calculates overlap matrix of vectors'''
        raise NotImplementedError
    
    @staticmethod
    def extendMatrixRepresentation(operator,vectors,opMat):
        ''' Extends the existing operator matrix representation (opMat) 
        with the elements of newly added vector
        (last member of the "vectors" list)

        out: Extended matrix representation (opMat)'''

        raise NotImplementedError
    
    @staticmethod
    def extendOverlapMatrix(vectors,overlap):
        ''' Extends the existing overlap matrix (overlap) 
        with the elements of newly added vector 
        (last member of the "vectors" list)

        out: Extended overlap matrix (overlap)'''
        
        raise NotImplementedError
