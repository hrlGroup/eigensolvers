""" Tree Tensor Network State (TTNS) vector wrapper"""
from __future__ import annotations # class returns class in typehint
import numpy as np
from abstractVector import AbstractVector, LINDEP_DEFAULT_VALUE
from typing import List, Optional, Dict
from numbers import Number

import warnings
import copy
from ttns2.state import TTNS
from ttns2.renormalization import AbstractRenormalization, SumOfOperators
from ttns2.sweepAlgorithms import LinearSystem, StateFitting
from ttns2.driver import bracket, getRenormalizedOp
from ttns2.driver import overlapMatrix as _overlapMatrix
from ttns2.driver import orthogonalizeAgainstSet

class TTNSVector(AbstractVector):
    def __init__(self, ttns: TTNS, options:Dict[str, Dict]):
        """ TTNSVector class

        `options` should be an optional dictionary of options containing:
            `sweepAlgorithmArgs` for default options for different sweepalgorithms
            `stateFittingArgs` overwrites `sweepAlgorithmArgs` for `StateFitting`
            `orthogonalizationArgs` overwrites `sweepAlgorithmArgs` for `Orthogonalization`
            `linearSystemArgs` overwrites `sweepAlgorithmArgs` for `LinearSystem`.
            `compressArgs` overwrites ``sweepAlgorithmArgs`` for `compress`.
        See the respective classes in `SweepAlgorithms` for the possible options.
        """
        self.ttns = ttns
        self.options = options
        # default options
        self.options["sweepAlgorithmArgs"] = options.get("sweepAlgorithmArgs", {"nSweep":1000, "convTol":1e-8})
        assert self.options["sweepAlgorithmArgs"] is not None
        op = self.options["sweepAlgorithmArgs"]
        # some changed default options
        op["indent"] = op.get("indent","\t")
        self.options["stateFittingArgs"] = options.get("stateFittingArgs", self.options["sweepAlgorithmArgs"])
        self.options["orthogonalizationArgs"] = options.get("orthogonalizationArgs", self.options["sweepAlgorithmArgs"])
        self.options["linearSystemArgs"] = options.get("linearSystemArgs", self.options["sweepAlgorithmArgs"])
        self.options["compressArgs"] = options.get("compressArgs", self.options["sweepAlgorithmArgs"])
        assert self.options["stateFittingArgs"] is not None
        assert self.options["orthogonalizationArgs"] is not None
        assert self.options["linearSystemArgs"] is not None
        assert self.options["compressArgs"] is not None

    @property
    def hasExactAddition(self):
        """
        Simplication of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        But does not exactly same as 2c for TTNS
        """
        return False

    @property
    def dtype(self):
        return np.result_type(*self.ttns.dtypes())

    @property
    def maxD(self) -> int:
        """Returns maximum value of virtual bond dimensions of a vectors.
        It is a wrapper function of ttns.maxD(), used for TTNSVectors"""
        return self.ttns.maxD()
   
    def __len__(self):
        raise NotImplementedError

    def __mul__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        warnings.warn("This copies the TTNS. This should be avoided! use inplace")
        new = self.copy()
        new.ttns.rootNode.tens *= other
        return new
    
    def __rmul__(self,other):
        raise NotImplementedError

    def __truediv__(self, other: Number) -> TTNSVector:
        warnings.warn("This copies the TTNS. This should be avoided!")
        new = self.copy()
        new.ttns.rootNode.tens /= other
        return new

    def __imul__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        self.ttns.rootNode.tens *= other
        return self

    def __itruediv__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        self.ttns.rootNode.tens /= other
        return self

    def normalize(self) -> TTNSVector:
        self.ttns.normalize()
        return self

    def norm(self) -> float:
        return self.ttns.norm()

    def real(self):
        raise NotImplementedError

    def conjugate(self:TTNSVector) -> TTNSVector:
        return TTNSVector(self.ttns.conj(),self.options)

    def vdot(self, other: TTNSVector, conjugate=True) -> Number:
        if not conjugate:
            # need to change RenormalizedDot accordingly
            raise NotImplementedError
        return bracket(self.ttns, other.ttns)

    def copy(self) -> TTNSVector:
        # ATTENTION: `options` should not be copied.
        #  Copying will lead to problems e.g. if the options contain `auxList`
        return TTNSVector(self.ttns.copy(), self.options)

    def applyOp(self, op: AbstractRenormalization) -> TTNSVector:
        warnings.warn("TTNS call to `applyOp`. This should be avoided!")
        # Need to add operators to `StateFitting`
        raise NotImplementedError

    def compress(self):
        """ Compresses bond dimension of TTNS"""
        
        args = self.options["compressArgs"]
        out = self.copy()
        solver = StateFitting(self.ttns, out.ttns, [1.0], **args)
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("compress: TTNS sweeps not converged!")
        return out

    @staticmethod
    def linearCombination(vectors: List[TTNSVector], coeffs:Optional[List[Number]]=None) -> TTNSVector:
        # Initial guess: The one with largest coefficient.
        if coeffs is not None:
            toOpt = vectors[np.argmax(np.abs(coeffs))].copy()
        else:
            norms = [o.norm() for o in vectors]
            toOpt = vectors[np.argmax(norms)].copy()
        solver = StateFitting([v.ttns for v in vectors], toOpt.ttns, coeffs,
                        **vectors[0].options["stateFittingArgs"])
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("linearCombination: TTNS sweeps not converged!")
        return toOpt

    @staticmethod
    def orthogonalize(xs,lindep = LINDEP_DEFAULT_VALUE) -> List[TTNSVector]:
        raise NotImplementedError

    @staticmethod
    def orthogonalize_against_set(x:TTNSVector, vectors:List[TTNSVector],
                                  lindep = LINDEP_DEFAULT_VALUE) -> TTNSVector|None:
        listVectors = [vector.ttns for vector in vectors]
        options = x.options.get("orthogonalizationArgs",{})
        x.ttns = orthogonalizeAgainstSet(x.ttns, listVectors,
                                         normalize=False,
                                         **options)
        if x.norm()**2 < lindep:
            # TODO would be better to just return what I have # not priority
            return None
        else:
            x.normalize()
            return x

    @staticmethod
    def solve(H, b:TTNSVector, sigma:Number,
              x0: Optional[TTNSVector]=None,
              opType = "her",reverseGF=False) -> TTNSVector:
        if x0 is None:
            # TODO think about best options.
            #   D=1 TTNS?
            # x0=b corresponds to residual of x0 = 0 (LHS x0 - b)
            # the sign does not matter
            x0 = b.copy()
        op = getRenormalizedOp(x0.ttns, H, x0.ttns)

        coeffs = [-1.0,1.0] if not reverseGF else [1.0,-1.0]

        if abs(sigma) > 1e-16:
            LHS = SumOfOperators([op, getRenormalizedOp(x0.ttns, sigma, x0.ttns)], coeffs=coeffs)
        else:
            LHS = op
        assert "lhsOpType" not in x0.options["linearSystemArgs"] # or just delete it in a copy of the dict
        solver = LinearSystem(x0.ttns if x0 is not None else None,
                              LHS,
                              b.ttns,
                              lhsOpType = opType,
                              **x0.options["linearSystemArgs"])
        converged, val = solver.run()
        if not converged:
            warnings.warn("solve: TTNS sweeps not converged!")
        return x0

    @staticmethod
    def matrixRepresentation(operator, vectors:List[TTNSVector]):
        # vv assuming that operator has the same dtype
        dtype = np.result_type(*[v.dtype for v in vectors])
        N = len(vectors)
        M = np.empty([N,N],dtype=dtype)
        for i in range(N):
            bra = vectors[i].ttns
            for j in range(i, N):
                ket = vectors[j].ttns
                val = getRenormalizedOp(bra, operator, ket).bracket()
                M[i, j] = val
                M[j, i] = np.conjugate(val)
        return M

    @staticmethod
    def overlapMatrix(vectors:List[TTNSVector]):
        ''' Calculates overlap matrix of tensor network states'''
        return _overlapMatrix([v.ttns for v in vectors])
    
    @staticmethod
    def extendMatrixRepresentation(operator, vectors:List[TTNSVector],opMat:np.ndarray):
        ''' Extends the existing operator matrix representation (opMat) 
        with the elements of newly added vector
        (last member of the "vectors" list)

        out: Extended matrix representation (opMat)'''

        dtype = np.result_type(*[v.dtype for v in vectors])
        m = len(vectors)
        elems = np.empty((1,m),dtype=dtype)
        bra = vectors[-1].ttns
        for i in range(m):
            ket = vectors[i].ttns
            elems[0,i] = getRenormalizedOp(bra, operator, ket).bracket()
        opMat = np.append(opMat,elems[:,:-1].conj(),axis=0)
        opMat = np.append(opMat,elems.T,axis=1)
        return opMat
 
    @staticmethod
    def extendOverlapMatrix(vectors:List[TTNSVector],overlap:np.ndarray):
        ''' Extends the existing overlap matrix (overlap) 
        with the elements of newly added vector 
        (last member of the "vectors" list)

        out: Extended overlap matrix (overlap)'''
        
        dtype = np.result_type(*[v.dtype for v in vectors])
        m = len(vectors)

        elems = np.empty((1,m),dtype=dtype)
        for i in range(m):
            elems[0,i] = vectors[i].vdot(vectors[-1],True)
        overlap = np.append(overlap,elems[:,:-1].conj(),axis=0)
        overlap = np.append(overlap,elems.T,axis=1)
        return overlap
