import os
import sys
import tempfile
import unittest

import numpy as np


try:
    import block2  
    import pyblock2 
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    BLOCK2_AVAILABLE = True
except ImportError:
    BLOCK2_AVAILABLE = False
    import warnings
    warnings.warn("block2 not available. unit test will be skipped")

from block2Vector import Block2Vector


FCIDUMP = "./FCIDUMP"


def _options(driver, prefix="B2TEST"):
    sweep = {"n_sweeps": 6, "tol": 1e-9, "noises": [0.0], "iprint": 0}
    return {
        "driver": driver,
        "tagPrefix": prefix,
        "sweepAlgorithmArgs": sweep,
        "linearSystemArgs": {
            "n_sweeps": 8,
            "tol": 1e-9,
            "noises": [0.0],
            "bra_bond_dims": [120] * 8,
            "bond_dims": [120] * 8,
            "thrds": [1e-10] * 8,
            "linear_max_iter": 2000,
            "iprint": 0,
        },
        "stateFittingArgs": {
            "n_sweeps": 6,
            "tol": 1e-10,
            "noises": [1e-5] * 2 + [1e-6] * 2 + [1e-7] * 2,
            "bra_bond_dims": [120] * 6,
            "iprint": 0,
        },
        "orthogonalizationArgs": {
            "n_sweeps": 6,
            "tol": 1e-10,
            "noises": [1e-5] * 2 + [1e-6] * 2 + [1e-7] * 2,
            "bra_bond_dims": [120] * 6,
            "iprint": 0,
        },
        "expectationArgs": {"iprint": 0},
    }


@unittest.skipUnless(BLOCK2_AVAILABLE, "block2 is not importable")
class TestBlock2Vector(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.driver = DMRGDriver(
            scratch=os.path.join(self.tmpdir.name, "block2_scratch"),
            symm_type=SymmetryTypes.SZ,
            n_threads=4,
        )
        self.driver.read_fcidump(filename=FCIDUMP, pg="d2h")
        self.driver.initialize_system(
            n_sites=self.driver.n_sites,
            n_elec=self.driver.n_elec,
            spin=self.driver.spin,
            orb_sym=self.driver.orb_sym,
        )
        self.driver.bw.b.Random.rand_seed(1234)
        self.mpo = self.driver.get_qc_mpo(
            h1e=self.driver.h1e,
            g2e=self.driver.g2e,
            ecore=self.driver.ecore,
            iprint=0,
        )
        self.options = _options(self.driver)

    def tearDown(self):
        self.driver.finalize()
        self.tmpdir.cleanup()

    def _random_vector(self, tag, bond_dim=60):
        mps = self.driver.get_random_mps(tag=tag, bond_dim=bond_dim, dot=2)
        return Block2Vector(mps, self.options)

    def test_core_operations(self):
        v1 = self._random_vector("V1")
        v2 = self._random_vector("V2")

        self.assertFalse(v1.hasExactAddition)
        self.assertEqual(v1.dtype, np.dtype(np.float64))
        self.assertGreater(v1.maxD, 0)
        np.testing.assert_allclose(v1.norm(), 1.0, atol=1e-12)

        copied = v1.copy()
        np.testing.assert_allclose(abs(copied.vdot(copied)), 1.0, atol=1e-8)
        np.testing.assert_allclose(abs(v1.vdot(copied)), 1.0, atol=1e-8)
        taggedCopy = v1.copy(tag="EXPLICIT-COPY-TAG")
        self.assertEqual(taggedCopy.mps.info.tag, "EXPLICIT-COPY-TAG")
        np.testing.assert_allclose(abs(v1.vdot(taggedCopy)), 1.0, atol=1e-8)

        scaled = 2.0 * v1
        np.testing.assert_allclose(abs(scaled.vdot(scaled)), 4.0, atol=1e-8)

        combo = Block2Vector.linearCombination([v1, v2], [1.0, -0.25])
        self.assertTrue(np.isfinite(combo.norm()))
        self.assertTrue(np.isfinite(combo.vdot(combo)))

        orthogonal = Block2Vector.orthogonalize_against_set(v2, [v1])
        self.assertIsNotNone(orthogonal)
        np.testing.assert_allclose(orthogonal.norm(), 1.0, atol=1e-8)
        np.testing.assert_allclose(v1.vdot(orthogonal), 0.0, atol=1e-6)

        hmat = Block2Vector.matrixRepresentation(self.mpo, [v1, orthogonal])
        smat = Block2Vector.overlapMatrix([v1, orthogonal])
        np.testing.assert_allclose(hmat, hmat.T, atol=1e-10)
        np.testing.assert_allclose(smat, np.eye(2), atol=1e-6)

        h_ext = Block2Vector.extendMatrixRepresentation(
            self.mpo, [v1, orthogonal], hmat[:1, :1]
        )
        s_ext = Block2Vector.extendOverlapMatrix([v1, orthogonal], smat[:1, :1])
        np.testing.assert_allclose(h_ext, hmat, atol=1e-10)
        np.testing.assert_allclose(s_ext, smat, atol=1e-10)

    def test_save_and_solve(self):
        vector = self._random_vector("SAVE-SOLVE")

        save_dir = os.path.join(self.tmpdir.name, "saved_block2_vector")
        vector.save(save_dir, additionalInformation={"outerIter": 2})
        self.assertTrue(os.path.isfile(os.path.join(save_dir, "mps_info.bin")))
        self.assertTrue(os.path.isfile(os.path.join(save_dir, "metadata.npz")))

        solution = Block2Vector.solve(self.mpo, vector, -2.0)
        self.assertTrue(np.isfinite(solution.norm()))
        self.assertGreater(solution.norm(), 0.0)
        shifted = -1.0 * self.mpo
        shifted.const_e += -2.0
        applied = solution.applyOp(shifted)
        residual = Block2Vector.linearCombination([applied, vector], [1.0, -1.0])
        self.assertLess(abs(residual.vdot(residual)) ** 0.5, 1e-4)

    def test_inexact_lanczos(self):
        from inexact_Lanczos import inexactLanczosDiagonalization
        from util_funcs import get_pick_function_close_to_sigma

        self.driver.bw.b.Random.rand_seed(1)
        guess = self._random_vector("LANCZOS-GUESS", bond_dim=80)
        guess.normalize()

        ev, vectors, status = inexactLanczosDiagonalization(
            self.mpo,
            guess,
            sigma=-2.0,
            L=4,
            maxit=2,
            eConv=1e-4,
            writeOut=False,
            saveAllVectors=False,
            pick=get_pick_function_close_to_sigma(-2.0),
            checkFitTol=1e-2,
        )

        self.assertTrue(status["isConverged"]) # TODO fails sometimes
        self.assertEqual(len(vectors), len(ev))
        self.assertLess(np.min(np.abs(ev + 1.975704)), 1e-2)
        np.testing.assert_allclose(
            Block2Vector.overlapMatrix(vectors), np.eye(len(vectors)), atol=1e-6
        )

if __name__ == "__main__":
    unittest.main()
