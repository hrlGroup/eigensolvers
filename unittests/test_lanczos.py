import unittest
import sys
from inexact_Lanczos  import (lowdinOrthoMatrix,diagonalizeHamiltonian,
        basisTransformation,inexactLanczosDiagonalization)
import numpy as np
from scipy import linalg as la
from numpyVector import NumpyVector
from util_funcs import find_nearest
import time
from util_funcs import get_pick_function_close_to_sigma

class Test_lanczos(unittest.TestCase):

    def setUp(self):
        n = 100
        ev = np.linspace(1,200,n)
        np.random.seed(1212)
        Q = la.qr(np.random.rand(n,n))[0]
        #Q = la.qr(np.random.rand(n,n)+1j*np.random.rand(n,n))[0]
        A = Q.T @ np.diag(ev) @ Q
        assert(la.ishermitian(A, atol=1e-08, rtol=1e-08))
        

        options = {"linearSolver":"gcrotmk","linearIter":1000,"linear_tol":1e-04}
        optionDict = {"linearSystemArgs":options}
        self.writeOut = False
        Y0 = NumpyVector(np.random.random((n)),optionDict)
        
        self.guess = Y0
        self.mat = A
        self.ev = ev                     
        self.sigma = 30
        self.eShift = 0.0
        self.L = 6
        self.maxit = 4
        self.eConv = 1e-6

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
        self.pick = get_pick_function_close_to_sigma(self.sigma)

        
    def test_lanczos(self):
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,self.sigma,self.L,
                self.maxit,self.eConv,pick=self.pick,writeOut=self.writeOut)
        typeClass = uvLanczos[0].__class__
        S = typeClass.overlapMatrix(uvLanczos)
        with self.subTest("returnType"):
            '''Check that the returned eigenvalues and eigenvectors have the correct types.'''
            self.assertIsInstance(evLanczos, np.ndarray)
            self.assertIsInstance(uvLanczos, list)
            self.assertIsInstance(uvLanczos[0], NumpyVector)
        with self.subTest("orthogonal"):
            '''Check that the returned basis is orthogonal.'''
            typeClass = uvLanczos[0].__class__
            np.testing.assert_allclose(S,np.eye(S.shape[0]),atol=1e-5)
        with self.subTest("transformationMatrix"):
            ''' XH@S@X = 1'''
            assert len(uvLanczos) > 1
            S1 = typeClass.overlapMatrix(uvLanczos)
            Hmat = typeClass.matrixRepresentation(self.mat,uvLanczos)
            uS = lowdinOrthoMatrix(S1,status)[1]
            ev,uv = diagonalizeHamiltonian(uS,Hmat)
            uSH = uS@uv
            mat = uSH.T.conj()@S@uSH
            np.testing.assert_allclose(mat,np.eye(mat.shape[0]),atol=1e-5)
        with self.subTest("extension"):
            '''Check that matrix extension works.'''
            Sfull = S
            S1 = typeClass.overlapMatrix(uvLanczos[:-1])
            _S = typeClass.extendOverlapMatrix(uvLanczos,S1)
            Hmatfull = typeClass.matrixRepresentation(self.mat,uvLanczos)
            Hmat1 = typeClass.matrixRepresentation(self.mat,uvLanczos[:-1])
            Hmat = typeClass.extendMatrixRepresentation(self.mat,uvLanczos,Hmat1)
            np.testing.assert_allclose(_S,Sfull,atol=1e-9)
            np.testing.assert_allclose(Hmat,Hmatfull,atol=1e-9)
        with self.subTest("eigenvalue"):
            '''Check that the calculated eigenvalue is accurate enough.'''
            target_value = find_nearest(evLanczos,self.sigma)[1]
            closest_value = find_nearest(self.ev,self.sigma)[1]        # comparing with exact value
            self.assertTrue((abs(target_value-closest_value)<= 1e-4),'Not accurate up to 1e-4')
        with self.subTest("eigenvector"):
            '''Check that the calculated eigenvector is accurate enough.'''
            idxE = find_nearest(self.evEigh,self.sigma)[0]
            idxT = find_nearest(evLanczos,self.sigma)[0]
            exactVector = self.uvEigh[:,idxE]
            lanczosVector = uvLanczos[idxT].array

            ovlp = np.vdot(exactVector,lanczosVector)
            np.testing.assert_allclose(abs(ovlp), 1, rtol=1e-5, err_msg = f"{ovlp=} but it should be +-1")
            lanczosVector = lanczosVector * ovlp
            np.testing.assert_allclose(exactVector,lanczosVector,rtol=1e-5,atol=1e-4)


if __name__ == '__main__':
    unittest.main()
