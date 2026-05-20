import unittest
import sys
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
        self.nc = 8            # number of contour points
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

    def test_feast(self):
        evfeast, uvfeast, status = feastDiagonalization(self.mat, self.guess, 
                self.nc, self.quad, self.rmin, self.rmax,self.eConv, self.maxit,
                writeOut=False)

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
        with self.subTest("eigenvalue"):
            """Check that the calculated eigenvalues are accurate enough."""
            with self.subTest("All contour eigenvalues"):
                contour_ev = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
                ncontour_ev = len(contour_ev)
                nfeast_ev = len(evfeast)
                # Account for the orthogonal basis case.
                self.assertTrue((ncontour_ev <= nfeast_ev),'All eigenvalues within contour must be calculated')

            with self.subTest("eigenvalue accuracy"):
                contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
                feast_evs = select_within_range(evfeast, self.rmin, self.rmax)[0]
                for i in range(len(contour_evs)):
                    target_value = contour_evs[i]
                    closest_value = find_nearest(feast_evs,target_value)[1]
                    self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 1e-4')

    def test_eigenvector(self):
        """Check that the calculated eigenvectors are accurate enough."""
       
        evfeast, uvfeast = feastDiagonalization(self.mat,self.guess,self.nc,self.quad,self.rmin,self.rmax,
                1e-12,40,writeOut=False)[0:2]
        
        contour_evs = select_within_range(self.evEigh, self.rmin, self.rmax)[0]
        for i in range(len(contour_evs)):
            idxE = find_nearest(self.evEigh,contour_evs[i])[0]
            idxT = find_nearest(evfeast,contour_evs[i])[0]
            exactVector = self.uvEigh[:,idxE]
            feastVector = uvfeast[idxT].array

            ovlp = np.vdot(exactVector,feastVector)
            # Test overlap; 0.99 is enough for this test.
            np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-2, err_msg = f"{ovlp=} but it should be +-1")
            feastVector = feastVector * ovlp
            np.testing.assert_allclose(exactVector,feastVector,rtol=1e-2,atol=1e-2)


if __name__ == '__main__':
    unittest.main()
