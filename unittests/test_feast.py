import unittest
import sys
import copy
import tempfile
from pathlib import Path
from feast  import *
from magic import ipsh
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time
from util_funcs import select_within_range

class Test_feast(unittest.TestCase):

    def setUp(self):
        n = 100
        ev = np.linspace(1,200,n)
        np.random.seed(10)
        Q = la.qr(np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        linOp = LinearOperator((n,n), matvec = lambda x, A=A: A@x)

        # Specify FEAST parameters
        self.rmin = 160.0
        self.rmax = 166.0
        self.n_quad = 8        # number of quadrature points
        self.quad = "legendre" # Choice of quadrature points
        m0 = 6                 # subspace dimension
        self.eConv = 1e-10      # residual convergence tolerance
        self.maxit = 20        # maximum FEAST iterations



        options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-02}
        optionDict = {"linearSystemArgs":options}
        
        Y0    = np.random.random((n,m0)) 
        for i in range(m0):
            Y0[:,i] = np.ones(n) * (i+1)
        Y1 = la.qr(Y0,mode="economic")[0]

        Y = []
        for i in range(m0):
            Y.append(NumpyVector(Y1[:,i], optionDict))

        self.guess = Y
        self.mat = A

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh

    def _subspaceConstructionCases(self, maxit):
        return [
                (1,"fitted_sums",maxit),
                (2,"double_sums",maxit),
                (3,"expanded_space",2),
                ]

    def _assertEigenvalues(self, evfeast):
        """Check that the calculated eigenvalues are accurate enough."""
        contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
        feast_evs = select_within_range(evfeast, self.rmin, self.rmax)[0]
        self.assertGreaterEqual(
                len(feast_evs),len(contour_evs),
                "All eigenvalues within contour must be calculated")

        for target_value in contour_evs:
            closest_value = find_nearest(feast_evs,target_value)[1]
            self.assertTrue(
                    (abs(target_value-closest_value)<= 1e-4),
                    "Not accurate up to 1e-4")

    def _assertEigenvectors(self, evfeast, uvfeast):
        """Check that the calculated eigenvectors are accurate enough."""
        contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
        for target_value in contour_evs:
            idxE = find_nearest(self.evEigh,target_value)[0]
            idxT = find_nearest(evfeast,target_value)[0]
            exactVector = self.uvEigh[:,idxE]
            feastVector = uvfeast[idxT].array

            ovlp = np.vdot(exactVector,feastVector)
            # Test overlap; 0.99 is enough for this test.
            np.testing.assert_allclose(
                    abs(ovlp), 1, rtol=1e-2,
                    err_msg = f"{ovlp=} but it should be +-1")
            feastVector = feastVector * ovlp
            np.testing.assert_allclose(
                    exactVector,feastVector,rtol=1e-2,atol=1e-2)

    def _assertSubspaceEigenvaluesMatch(self, reference, candidate):
        targets = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
        reference = select_within_range(reference, self.rmin, self.rmax)[0]
        candidate = select_within_range(candidate, self.rmin, self.rmax)[0]
        for target_value in targets:
            reference_value = find_nearest(reference,target_value)[1]
            candidate_value = find_nearest(candidate,target_value)[1]
            np.testing.assert_allclose(
                    candidate_value,reference_value,rtol=1e-7,atol=1e-4)

    def test_feast(self):
        results = {}
        for subspaceConstruction,expected,maxit in self._subspaceConstructionCases(self.maxit):
            with self.subTest(subspaceConstruction=expected):
                evfeast, uvfeast, status = feastDiagonalization(
                        self.mat, copy.deepcopy(self.guess),
                        self.n_quad, self.quad, self.rmin, self.rmax,
                        self.eConv, maxit, writeOut=False,
                        subspaceConstruction=subspaceConstruction)

                self.assertEqual(status["subspaceConstruction"], expected)
                self.assertIsInstance(evfeast, np.ndarray)
                self.assertIsInstance(uvfeast, list)
                self.assertIsInstance(uvfeast[0], NumpyVector)
                self._assertEigenvalues(evfeast)
                results[expected] = evfeast, uvfeast, status

        evfeast, uvfeast, status = results["fitted_sums"]

        with self.subTest("returnType"):
            """Check that the returned eigenvalues and eigenvectors have the correct types."""
            self.assertIsInstance(evfeast, np.ndarray)
            self.assertIsInstance(uvfeast, list)
            self.assertIsInstance(uvfeast[0], NumpyVector)
        with self.subTest("eigenvalue"):
            """Check that the calculated eigenvalues are accurate enough."""
            with self.subTest("All contour eigenvalues"):
                contour_ev = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
                ncontour_ev = len(contour_ev)
                nfeast_ev = len(evfeast)
                # Account for the orthogonal basis case.
                self.assertTrue((ncontour_ev <= nfeast_ev), 'All eigenvalues within contour must be calculated')
            with self.subTest("eigenvalue accuracy"):
                contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
        with self.subTest("back-transform"):
            """Check the linear combination."""
            typeClass = uvfeast[0].__class__
            S = typeClass.overlapMatrix(uvfeast[:-1])
            assert len(uvfeast) > 1
            Hmat = typeClass.matrixRepresentation(self.mat,uvfeast[:-1])
            SmatFull = typeClass.overlapMatrix(uvfeast)
            uS = lowdinOrthoMatrix(SmatFull,status)[1]
            HmatFull = typeClass.matrixRepresentation(self.mat,uvfeast)
            uv = diagonalizeHamiltonian(uS,HmatFull)[1]
            uSH = uS@uv
            bases = basisTransformation(uvfeast,uSH)
            for m in range(len(uvfeast)):
                ovlp = bases[m].vdot(uvfeast[m],True)
                np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg
                        = f"{ovlp=} but it should be +-1")
                np.testing.assert_allclose(uvfeast[m].array,ovlp*bases[m].array,atol=1e-5)
        with self.subTest("orthonormal"):
            """Check that the returned basis is orthogonal."""
            typeClass = uvfeast[0].__class__
            S = typeClass.overlapMatrix(uvfeast)
            np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5)
            with self.subTest("transformationMatrix"):
                """ XH@S@X = 1"""
                typeClass = uvfeast[0].__class__
                assert len(uvfeast) > 1
                uS = lowdinOrthoMatrix(SmatFull,status)[1]
                HmatFull = typeClass.matrixRepresentation(self.mat,uvfeast)
                uv = diagonalizeHamiltonian(uS,HmatFull)[1]
                uSH = uS@uv
                mat = uSH.T.conj()@S@uSH
                np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5)
    def test_eigenvector(self):
        """Check that the calculated eigenvectors are accurate enough."""
        for subspaceConstruction,expected,maxit in self._subspaceConstructionCases(40):
            with self.subTest(subspaceConstruction=expected):
                evfeast, uvfeast = feastDiagonalization(
                        self.mat,copy.deepcopy(self.guess),self.n_quad,self.quad,
                        self.rmin,self.rmax,1e-12,maxit,writeOut=False,
                        subspaceConstruction=subspaceConstruction)[0:2]
                self._assertEigenvectors(evfeast,uvfeast)

    def test_subspace_construction_options(self):
        """Check the three FEAST subspace construction options."""
        ev_fit, uv_fit, status_fit = feastDiagonalization(
                self.mat,self.guess,self.n_quad,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit,writeOut=False,subspaceConstruction=1)
        ev_double, uv_double, status_double = feastDiagonalization(
                self.mat,self.guess,self.n_quad,self.quad,self.rmin,self.rmax,
                self.eConv,self.maxit,writeOut=False,subspaceConstruction=2)
        ev_expanded, uv_expanded, status_expanded = feastDiagonalization(
                self.mat,self.guess,self.n_quad,self.quad,self.rmin,self.rmax,
                self.eConv,2,writeOut=False,subspaceConstruction=3)

        self.assertEqual(status_fit["subspaceConstruction"],"fitted_sums")
        self.assertEqual(status_double["subspaceConstruction"],"double_sums")
        self.assertEqual(status_expanded["subspaceConstruction"],"expanded_space")
        np.testing.assert_allclose(ev_fit,ev_double,rtol=1e-7,atol=1e-7)
        self._assertSubspaceEigenvaluesMatch(ev_fit,ev_expanded)
        self.assertEqual(len(ev_fit),len(uv_fit))
        self.assertEqual(len(ev_double),len(uv_double))
        self.assertEqual(len(ev_expanded),len(uv_expanded))
        self.assertGreater(len(ev_expanded),len(ev_fit))

    def test_save_all_vectors(self):
        """Check saving FEAST vectors with eigenvalues and coefficients."""
        with tempfile.TemporaryDirectory() as saveDir:
            evfeast, uvfeast, status = feastDiagonalization(
                    self.mat,copy.deepcopy(self.guess),
                    self.n_quad,self.quad,self.rmin,self.rmax,
                    self.eConv,1,writeOut=False,
                    saveAllVectors=True,saveDir=saveDir)

            files = sorted(Path(saveDir).glob("vector_000_*.npz"))
            self.assertEqual(len(files),len(self.guess))
            with np.load(files[0],allow_pickle=True) as data:
                self.assertIn("vector",data.files)
                self.assertIn("eigencoefficients",data.files)
                self.assertIn("eigenvalues",data.files)
                self.assertIn("status",data.files)
                self.assertEqual(data["eigencoefficients"].shape[0],len(files))
                self.assertEqual(data["eigencoefficients"].shape[1],len(data["eigenvalues"]))
                np.testing.assert_allclose(data["eigenvalues"],evfeast)
                self.assertEqual(
                        data["status"].item()["subspaceConstruction"],
                        status["subspaceConstruction"])

if __name__ == '__main__':
    unittest.main()
