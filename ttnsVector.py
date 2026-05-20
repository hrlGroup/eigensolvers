"""Tree Tensor Network State (TTNS) vector wrapper."""
from __future__ import annotations # allow class names in type hints
import numpy as np
from abstractVector import AbstractVector, LINDEP_DEFAULT_VALUE
from typing import List, Optional, Dict
from numbers import Number

import warnings
import copy
from pathlib import Path
from ttns2.state import TTNS
from ttns2.renormalization import AbstractRenormalization, SumOfOperators
from ttns2.sweepAlgorithms import LinearSystem, StateFitting
from ttns2.driver import bracket, getRenormalizedOp
from ttns2.driver import overlapMatrix as _overlapMatrix
from ttns2.driver import orthogonalizeAgainstSet

class TTNSVector(AbstractVector):
    def __init__(self, ttns: TTNS, options:Dict[str, Dict]):
        """ TTNSVector class

        `options` is a dictionary that may contain:
            `sweepAlgorithmArgs`: default options for different sweep algorithms
            `stateFittingArgs`: overrides `sweepAlgorithmArgs` for `StateFitting`
            `orthogonalizationArgs`: overrides `sweepAlgorithmArgs` for `Orthogonalization`
            `linearSystemArgs`: overrides `sweepAlgorithmArgs` for `LinearSystem`
            `compressArgs`: overrides `sweepAlgorithmArgs` for `compress`
        See the respective classes in `SweepAlgorithms` for available options.
        """
        self.ttns = ttns
        self.options = options
        # Default options.
        self.options["sweepAlgorithmArgs"] = options.get("sweepAlgorithmArgs", {"nSweep":1000, "convTol":1e-8})
        assert self.options["sweepAlgorithmArgs"] is not None
        op = self.options["sweepAlgorithmArgs"]
        # Adjust selected defaults.
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
        Simplification of vector addition with its complex conjugate.
        For example, c+c* = 2c when c=(a+ib)
        This summation is true for numpy vectors
        but is not exactly identical to 2c for TTNS.
        """
        return False

    @property
    def dtype(self):
        return np.result_type(*self.ttns.dtypes())

    @property
    def maxD(self) -> int:
        """Return the maximum virtual bond dimension of a vector.
        This wraps ttns.maxD() for TTNSVector."""
        return self.ttns.maxD()
   
    def __len__(self):
        raise NotImplementedError

    def __mul__(self, other: Number) -> TTNSVector:
        assert isinstance(other, Number)
        warnings.warn("This copies the TTNS. Prefer in-place scaling.")
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
            # RenormalizedDot would need to be changed accordingly.
            raise NotImplementedError
        return bracket(self.ttns, other.ttns)

    def copy(self) -> TTNSVector:
        # ATTENTION: `options` should not be copied.
        #  Copying will lead to problems e.g. if the options contain `auxList`
        return TTNSVector(self.ttns.copy(), self.options)

    def save(self, filename, additionalInformation:dict=None):
        filename = Path(filename)
        if filename.suffix == "":
            filename = filename.with_suffix(".h5")
        self.ttns.saveToHDF5(str(filename), additionalInformation=additionalInformation)

    def applyOp(self, op: AbstractRenormalization) -> TTNSVector:
        warnings.warn("TTNS call to `applyOp`. This should be avoided!")
        # Need to add operators to `StateFitting`
        raise NotImplementedError

    def compress(self):
        """Compress the bond dimension of a TTNS."""
        
        args = self.options["compressArgs"]
        out = self.copy()
        solver = StateFitting(self.ttns, out.ttns, [1.0], **args)
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("compress: TTNS sweeps did not converge!")
        return out

    @staticmethod
    def linearCombination(vectors: List[TTNSVector], coeffs:Optional[List[Number]]=None) -> TTNSVector:
        # Initial guess: use the vector with the largest coefficient.
        if coeffs is not None:
            toOpt = vectors[np.argmax(np.abs(coeffs))].copy()
        else:
            norms = [o.norm() for o in vectors]
            toOpt = vectors[np.argmax(norms)].copy()
        solver = StateFitting([v.ttns for v in vectors], toOpt.ttns, coeffs,
                        **vectors[0].options["stateFittingArgs"])
        converged, optVal = solver.run()
        if not converged:
            warnings.warn("linearCombination: TTNS sweeps did not converge!")
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
            # TODO: It may be better to return the current vector.
            return None
        else:
            x.normalize()
            return x

    @staticmethod
    def solve(H, b:TTNSVector, sigma:Number,
              x0: Optional[TTNSVector]=None,
              opType = "her",reverseGF=False) -> TTNSVector:
        if x0 is None:
            # TODO: decide on the best default initial guess.
            #   D=1 TTNS?
            # x0=b corresponds to the residual of x0 = 0 (LHS x0 - b).
            # The sign does not matter.
            x0 = b.copy()
        op = getRenormalizedOp(x0.ttns, H, x0.ttns)

        coeffs = [-1.0,1.0] if not reverseGF else [1.0,-1.0]

        if abs(sigma) > 1e-16:
            LHS = SumOfOperators([op, getRenormalizedOp(x0.ttns, sigma, x0.ttns)], coeffs=coeffs)
        else:
            LHS = op
        assert "lhsOpType" not in x0.options["linearSystemArgs"] # or delete it in a copy of the dict
        solver = LinearSystem(x0.ttns if x0 is not None else None,
                              LHS,
                              b.ttns,
                              lhsOpType = opType,
                              **x0.options["linearSystemArgs"])
        converged, val = solver.run()
        if not converged:
            warnings.warn("solve: TTNS sweeps did not converge!")
        return x0

    @staticmethod
    def matrixRepresentation(operator, vectors:List[TTNSVector]):
        # Assume that the operator has the same dtype.
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
        """Calculate the overlap matrix of tensor network states."""
        return _overlapMatrix([v.ttns for v in vectors])
    
    @staticmethod
    def extendMatrixRepresentation(operator, vectors:List[TTNSVector],opMat:np.ndarray):
        """Extend the existing operator matrix representation (opMat)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended matrix representation (opMat)"""

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
        """Extend the existing overlap matrix (overlap)
        with the elements of the newly added vector
        (last member of the "vectors" list)

        out: Extended overlap matrix (overlap)"""
        
        dtype = np.result_type(*[v.dtype for v in vectors])
        m = len(vectors)

        elems = np.empty((1,m),dtype=dtype)
        for i in range(m):
            elems[0,i] = vectors[i].vdot(vectors[-1],True)
        overlap = np.append(overlap,elems[:,:-1].conj(),axis=0)
        overlap = np.append(overlap,elems.T,axis=1)
        return overlap
