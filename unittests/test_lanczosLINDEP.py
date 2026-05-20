import unittest
from inexact_Lanczos  import *
import numpy as np
from numpyVector import NumpyVector
    
class Test_lanczos(unittest.TestCase):
    def setUp(self):
        # The initial vector only has support on two eigenvectors, so the
        # Krylov space must run out of new directions at the second step.
        A = np.diag([1.0, 3.0, 10.0])
        options = {"linearSolver":"gcrotmk","linearIter":1000,
                   "linear_tol":1e-12, "linear_atol":1e-12}
        optionDict = {"linearSystemArgs":options}
        Y0 = NumpyVector(np.array([1.0, 1.0, 0.0]), optionDict)
        
        self.guess = Y0
        self.mat = A
        self.sigma = 2.0
        self.L = 5
        self.maxit = 5
        self.eConv = 1e-14

    def test_status(self):
        """Check that linear dependency is reported in the returned status."""
        evLanczos, uvLanczos, status = inexactLanczosDiagonalization(self.mat,self.guess,self.sigma,
                self.L,self.maxit,self.eConv,pick=None,writeOut=False,
                saveTNSsEachIteration=False)
        self.assertTrue(status["lindep"])
        # After a dependent vector is found, the returned vector list should
        # only contain the independent Krylov vectors.
        iKrylov = status["innerIter"]
        nvectors = len(uvLanczos)
        self.assertTrue(nvectors == iKrylov)

    def test_futileRestarts(self):
        """Check that ineffective restarts are counted."""
        status = {"lindep": True, "ref": [np.array([1.0])],
                  "futileRestarts": 0}
        decision = terminateRestart(np.array([2.0]), self.eConv, status)
        self.assertFalse(decision)
        self.assertEqual(status["futileRestarts"], 1)

        status["futileRestarts"] = 3
        with self.assertWarns(UserWarning):
            decision = terminateRestart(np.array([2.0]), self.eConv, status)
        self.assertTrue(decision)
        self.assertEqual(status["futileRestarts"], 4)

if __name__ == "__main__":
    unittest.main()
