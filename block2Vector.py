"""block2 MPS vector wrapper.

Currently does not support complex mode, so won't work with FEAST"""
from __future__ import annotations

import os
import shutil
import tempfile
from numbers import Number
from pathlib import Path
from typing import List, Optional

import numpy as np

from abstractVector import AbstractVector, LINDEP_DEFAULT_VALUE


_DEFAULT_SWEEP_ARGS = {
    "n_sweeps": 10,
    "tol": 1e-8,
    "noises": [1e-5] * 2 + [1e-6] * 2 + [1e-7] * 2,
    "iprint": 0,
}

_MULTIPLY_ARGS = {
    "n_sweeps",
    "tol",
    "bond_dims",
    "bra_bond_dims",
    "noises",
    "noise_mpo",
    "thrds",
    "left_mpo",
    "cutoff",
    "twosite_to_onesite",
    "linear_max_iter",
    "linear_rel_conv_thrd",
    "proj_mpss",
    "proj_weights",
    "proj_bond_dim",
    "solver_type",
    "right_weight",
    "iprint",
    "left_kernel",
    "right_kernel",
}

_MULTI_ADDITION_ARGS = {
    "n_sweeps",
    "tol",
    "bra_bond_dims",
    "ket_bond_dimss",
    "noises",
    "noise_mpo",
    "cutoff",
    "iprint",
}

_EXPECTATION_ARGS = {
    "stacked_mpo",
    "store_bra_spectra",
    "store_ket_spectra",
    "iprint",
}


def _filtered_args(args, allowed):
    """Return a copy of `args` containing only keys supported by a block2 call."""
    return {key: value for key, value in args.items() if key in allowed}


def _disable_noises_without_noise_mpo(args):
    if "noise_mpo" in args and args["noise_mpo"] is not None:
        return args
    noises = args.get("noises", None)
    if noises is not None and len(noises) > 0 and np.any(np.asarray(noises) != 0.0):
        args = dict(args)
        args["noises"] = [0.0]
    return args


class Block2Vector(AbstractVector):
    def __init__(self, mps, options: dict):
        """Wrap a block2 MPS in the AbstractVector interface.

        `options` must contain a block2 `DMRGDriver` under the key `driver`.
        Other option dictionaries mirror `TTNSVector`: `sweepAlgorithmArgs`,
        `stateFittingArgs`, `orthogonalizationArgs`, `linearSystemArgs`,
        `compressArgs`, `expectationArgs`, and `saveArgs`.
        """
        self.mps = mps
        self.options = options
        if "driver" not in options:
            raise KeyError("Block2Vector requires options['driver'].")

        self.options["sweepAlgorithmArgs"] = options.get(
            "sweepAlgorithmArgs", dict(_DEFAULT_SWEEP_ARGS)
        )
        sweepArgs = self.options["sweepAlgorithmArgs"]
        self.options["stateFittingArgs"] = options.get("stateFittingArgs", sweepArgs)
        self.options["orthogonalizationArgs"] = options.get(
            "orthogonalizationArgs", self.options["stateFittingArgs"]
        )
        self.options["linearSystemArgs"] = options.get("linearSystemArgs", sweepArgs)
        self.options["compressArgs"] = options.get("compressArgs", {})
        self.options["expectationArgs"] = options.get("expectationArgs", {"iprint": 0})
        self.options["saveArgs"] = options.get("saveArgs", {})
        self.options["tagPrefix"] = options.get("tagPrefix", "B2V")

    def _driver(self):
        return self.options["driver"]

    def _new_tag(self, label="TMP"):
        counter = self.options.get("_block2VectorTagCounter", 0)
        # NOTE this only works if self.options is kept as pointer throughout, 
        # which is the case right now
        self.options["_block2VectorTagCounter"] = counter + 1
        return f"{self.options['tagPrefix']}-{label}-{counter}"

    def _new_mps_like(self, label="TMP", bond_dim=None):
        if bond_dim is None:
            bond_dim = max(self.mps.info.bond_dim, self.maxD)
        return self._driver().get_random_mps(
            tag=self._new_tag(label),
            bond_dim=bond_dim,
            center=self.mps.center,
            dot=self.mps.dot,
            target=self.mps.info.target,
        )

    @staticmethod
    def _assert_real_driver(driver):
        try:
            from pyblock2.driver.core import SymmetryTypes

            if SymmetryTypes.CPX in driver.bw.symm_type:
                raise AssertionError("Block2Vector currently assumes real block2 data.")
        except ImportError:
            return
        except TypeError:
            return

    @staticmethod
    def _assert_real_scalar(value):
        if abs(np.imag(value)) > 1e-16:
            raise AssertionError("Block2Vector currently supports only real scalars.")
        return float(np.real(value))

    @staticmethod
    def _save_mps_info(driver, mps):
        mps.info.save_data(driver.scratch + "/%s-mps_info.bin" % mps.info.tag)

    @staticmethod
    def _copy_file(source, target):
        try:
            shutil.copy2(source, target)
        except Exception:
            shutil.copyfile(source, target)

    @staticmethod
    def copy_saved_file(save_dir, target_filename):
        source = Path(save_dir) / os.path.basename(target_filename)
        Block2Vector._copy_file(source, target_filename)

    @staticmethod
    def load_mps_from_dir(driver, save_dir, tag=None):
        saveDir = Path(save_dir)
        if tag is not None:
            infoFile = saveDir / f"{tag}-mps_info.bin"
        else:
            infoFile = saveDir / "mps_info.bin"
        mpsInfo = driver.bw.brs.MPSInfo(0)
        mpsInfo.load_data(str(infoFile))
        for iSite in range(mpsInfo.n_sites + 1):
            Block2Vector.copy_saved_file(save_dir, mpsInfo.get_filename(False, iSite))
            Block2Vector.copy_saved_file(save_dir, mpsInfo.get_filename(True, iSite))
        mpsInfo.load_mutable()
        mps = driver.bw.bs.MPS(mpsInfo)
        for iSite in range(-1, mpsInfo.n_sites):
            Block2Vector.copy_saved_file(save_dir, mps.get_filename(iSite))
        mps.load_data()
        mps.load_mutable()
        mpsInfo.bond_dim = mps.info.get_max_bond_dimension()
        return mps

    @staticmethod
    def _load_center_tensor(mps):
        mps.info.load_mutable()
        mps.load_mutable()
        if mps.tensors[mps.center] is None:
            mps.load_data()
        return mps.tensors[mps.center]

    @staticmethod
    def _scale_mps(driver, mps, scalar):
        scalar = Block2Vector._assert_real_scalar(scalar)
        tensor = Block2Vector._load_center_tensor(mps)
        tensor.data *= scalar
        mps.save_mutable()
        mps.info.save_mutable()
        mps.save_data()
        Block2Vector._save_mps_info(driver, mps)

    @property
    def hasExactAddition(self):
        return False

    @property
    def dtype(self):
        self._assert_real_driver(self._driver())
        return np.dtype(np.float64)

    @property
    def maxD(self) -> int:
        return self.mps.info.get_max_bond_dimension()

    def __len__(self):
        raise NotImplementedError

    def __mul__(self, other: Number) -> Block2Vector:
        assert isinstance(other, Number)
        new = self.copy()
        self._scale_mps(self._driver(), new.mps, other)
        return new

    def __rmul__(self, other: Number) -> Block2Vector:
        return self.__mul__(other)

    def __truediv__(self, other: Number) -> Block2Vector:
        assert isinstance(other, Number)
        new = self.copy()
        self._scale_mps(self._driver(), new.mps, 1.0 / other)
        return new

    def __imul__(self, other: Number) -> Block2Vector:
        assert isinstance(other, Number)
        self._scale_mps(self._driver(), self.mps, other)
        return self

    def __itruediv__(self, other: Number) -> Block2Vector:
        assert isinstance(other, Number)
        self._scale_mps(self._driver(), self.mps, 1.0 / other)
        return self

    def normalize(self) -> Block2Vector:
        tensor = self._load_center_tensor(self.mps)
        tensor.normalize()
        self.mps.save_mutable()
        self.mps.info.save_mutable()
        self.mps.save_data()
        self._save_mps_info(self._driver(), self.mps)
        return self

    def norm(self) -> float:
        tensor = self._load_center_tensor(self.mps)
        return float(np.linalg.norm(np.asarray(tensor.data)))

    def real(self) -> Block2Vector:
        self._assert_real_driver(self._driver())
        return self.copy()

    def conjugate(self) -> Block2Vector:
        self._assert_real_driver(self._driver())
        return self.copy()

    def vdot(self, other: Block2Vector, conjugate=True):
        if not conjugate:
            self._assert_real_driver(self._driver())
        driver = self._driver()
        args = _filtered_args(self.options["expectationArgs"], _EXPECTATION_ARGS)
        return driver.expectation(
            self.mps, driver.get_identity_mpo(), other.mps, **args
        )

    def copy(self, tag=None) -> Block2Vector:
        driver = self._driver()
        saveDir = tempfile.mkdtemp(prefix="block2VectorCopy-")
        self.save(saveDir)
        copiedMps = self.load_mps_from_dir(driver, saveDir, tag=self.mps.info.tag)
        copiedMps.info.tag = self._new_tag() if tag is None else tag
        copiedMps.save_mutable()
        copiedMps.info.save_mutable()
        copiedMps.save_data()
        self._save_mps_info(driver, copiedMps)
        return Block2Vector(copiedMps, self.options)

    def save(self, filename, additionalInformation: dict = None):
        filename = Path(filename)
        saveDir = filename if filename.suffix == "" else filename.with_suffix("")
        saveDir.mkdir(parents=True, exist_ok=True)

        self.mps.save_data()
        self.mps.info.save_data(str(saveDir / f"{self.mps.info.tag}-mps_info.bin"))
        self.mps.info.save_data(str(saveDir / "mps_info.bin"))

        for iSite in range(self.mps.n_sites + 1):
            source = self.mps.info.get_filename(False, iSite)
            self._copy_file(source, saveDir / os.path.basename(source))
            source = self.mps.info.get_filename(True, iSite)
            self._copy_file(source, saveDir / os.path.basename(source))
        for iSite in range(-1, self.mps.n_sites):
            source = self.mps.get_filename(iSite)
            self._copy_file(source, saveDir / os.path.basename(source))

        metadata = {
            "tag": self.mps.info.tag,
            "scratch": self._driver().scratch,
            "bond_dim": self.maxD,
        }
        if additionalInformation is not None:
            metadata.update(additionalInformation)
        np.savez(str(saveDir / "metadata.npz"), **metadata)

    def applyOp(self, op) -> Block2Vector:
        driver = self._driver()
        out = self.copy()
        args = _filtered_args(self.options["linearSystemArgs"], _MULTIPLY_ARGS)
        args.pop("left_mpo", None)
        args = _disable_noises_without_noise_mpo(args)
        driver.multiply(out.mps, op, self.mps, **args)
        return out

    def compress(self) -> Block2Vector:
        out = self.copy()
        args = self.options["compressArgs"]
        maxBondDim = args.get("max_bond_dim", None)
        out._driver().compress_mps(out.mps, max_bond_dim=maxBondDim)
        return out

    @staticmethod
    def _linearCombinationWithArgs(
        vectors: List[Block2Vector], coeffs: Optional[List[Number]], args
    ) -> Block2Vector:
        if coeffs is None:
            coeffs = [1.0] * len(vectors)
        if len(vectors) != len(coeffs):
            raise ValueError("vectors and coeffs must have the same length.")
        if len(vectors) == 0:
            raise ValueError("At least one vector is required.")

        coeffs = [Block2Vector._assert_real_scalar(coeff) for coeff in coeffs]
        if len(vectors) == 1:
            return coeffs[0] * vectors[0]

        driver = vectors[0]._driver()
        if any(vector._driver() is not driver for vector in vectors):
            raise ValueError("All Block2Vector objects must use the same driver.")

        guessIndex = int(np.argmax(np.abs(coeffs)))
        maxBondDim = max(vector.maxD for vector in vectors)
        out = Block2Vector(vectors[guessIndex]._new_mps_like("LC", maxBondDim), vectors[0].options)
        block2Args = _filtered_args(args, _MULTI_ADDITION_ARGS)
        block2Args = _disable_noises_without_noise_mpo(block2Args)
        driver.multi_addition(
            out.mps, [vector.mps for vector in vectors], mpos=coeffs, **block2Args
        )
        return out

    @staticmethod
    def linearCombination(
        vectors: List[Block2Vector], coeffs: Optional[List[Number]] = None
    ) -> Block2Vector:
        return Block2Vector._linearCombinationWithArgs(
            vectors, coeffs, vectors[0].options["stateFittingArgs"]
        )

    @staticmethod
    def orthogonalize(xs, lindep=LINDEP_DEFAULT_VALUE) -> List[Block2Vector]:
        raise NotImplementedError

    @staticmethod
    def orthogonalize_against_set(
        x: Block2Vector, vectors: List[Block2Vector], lindep=LINDEP_DEFAULT_VALUE
    ) -> Block2Vector | None:
        if len(vectors) == 0:
            x.normalize()
            return x

        coeffs = [1.0] + [-vector.vdot(x) for vector in vectors]
        out = Block2Vector._linearCombinationWithArgs(
            [x] + list(vectors), coeffs, x.options["orthogonalizationArgs"]
        )
        if out.norm() ** 2 < lindep:
            return None
        out.normalize()
        return out

    @staticmethod
    def solve(H, b: Block2Vector, sigma: Number, x0=None, opType="her", reverseGF=False):
        del opType
        sigma = Block2Vector._assert_real_scalar(sigma)
        driver = b._driver()
        Block2Vector._assert_real_driver(driver)
        x = b.copy() if x0 is None else x0.copy()

        if reverseGF:
            leftMpo = 1.0 * H
            leftMpo.const_e -= sigma
        else:
            leftMpo = -1.0 * H
            leftMpo.const_e += sigma
        if not hasattr(leftMpo, "const_e"):
            raise AttributeError("Shifted block2 MPO must expose const_e.")

        args = _filtered_args(b.options["linearSystemArgs"], _MULTIPLY_ARGS)
        args["left_mpo"] = leftMpo
        driver.multiply(x.mps, driver.get_identity_mpo(), b.mps, **args)
        return x

    @staticmethod
    def matrixRepresentation(operator, vectors: List[Block2Vector]):
        # Assume Hermitian operator
        dtype = np.result_type(*[vector.dtype for vector in vectors])
        driver = vectors[0]._driver()
        args = _filtered_args(vectors[0].options["expectationArgs"], _EXPECTATION_ARGS)
        nVectors = len(vectors)
        mat = np.empty((nVectors, nVectors), dtype=dtype)
        for i in range(nVectors):
            for j in range(i, nVectors):
                value = driver.expectation(
                    vectors[i].mps, operator, vectors[j].mps, **args
                )
                mat[i, j] = value
                mat[j, i] = np.conjugate(value)
        return mat

    @staticmethod
    def overlapMatrix(vectors: List[Block2Vector]):
        driver = vectors[0]._driver()
        return Block2Vector.matrixRepresentation(driver.get_identity_mpo(), vectors)

    @staticmethod
    def extendMatrixRepresentation(operator, vectors: List[Block2Vector], opMat):
        dtype = np.result_type(*[vector.dtype for vector in vectors])
        driver = vectors[0]._driver()
        args = _filtered_args(vectors[0].options["expectationArgs"], _EXPECTATION_ARGS)
        nVectors = len(vectors)
        elems = np.empty((1, nVectors), dtype=dtype)
        bra = vectors[-1].mps
        for i in range(nVectors):
            elems[0, i] = driver.expectation(bra, operator, vectors[i].mps, **args)
        opMat = np.append(opMat, elems[:, :-1].conj(), axis=0)
        opMat = np.append(opMat, elems.T, axis=1)
        return opMat

    @staticmethod
    def extendOverlapMatrix(vectors: List[Block2Vector], overlap):
        driver = vectors[0]._driver()
        return Block2Vector.extendMatrixRepresentation(
            driver.get_identity_mpo(), vectors, overlap
        )
