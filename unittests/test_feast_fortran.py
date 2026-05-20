import unittest
import numpy as np
import math
from pathlib import Path
from magic import ipsh
from numpyVector import NumpyVector
from util_funcs import quadraturePointsWeights
from feast  import calculateQuadrature, updateQ

# This tests our FEAST code (partial comparison) 
# with outputs of FEAST Fortran code (by Eric Polizzi)

filename = Path(__file__).with_name("data_fortranCode.out")

def read_fortranData(k=0):
    amat = np.loadtxt(filename, dtype = float,skiprows =1, max_rows=4)
    guess = np.loadtxt(filename,dtype =complex,skiprows=7, max_rows=3)
    xe = np.loadtxt(filename,dtype =float,skiprows=12, max_rows=8)
    we = np.loadtxt(filename,dtype =float,skiprows=22, max_rows=8)
    theta = np.loadtxt(filename,dtype =float,skiprows=32, max_rows=8)
    zne = np.loadtxt(filename,dtype =complex,skiprows=42, max_rows=8)
    
    Qe = np.loadtxt(filename,dtype =complex,skiprows=62+k*5, max_rows=3)
    Q = np.loadtxt(filename,dtype =float,skiprows=102+k*5, max_rows=3)
    return amat,guess,xe,we,theta,zne,Qe,Q

class Test_feast_fortran(unittest.TestCase):

    def setUp(self):
        A = read_fortranData()[0]
        n = A.shape[0]
        self.rmin = 3.0
        self.rmax = 5.0
        self.nc = 8            # number of contour points
        self.quad = "legendre" # Choice of quadrature points
        m0 = 3                 # subspace dimension
        self.eConv = 1e-12      # residual convergence tolerance
        self.maxit = 10        # maximum FEAST iterations
        self.efactor = 0.3
        self.order = [4,3,5,2,6,1,7,0]
        
        options = {"linearSolver":"pardiso"}
        optionsDict = {"linearSystemArgs":options}
        
        Y1 = read_fortranData()[1]
        Y = []
        for i in range(m0):
            Y.append(NumpyVector(Y1[i,:], optionsDict))

        self.guess = Y
        self.mat = A

        evEigh, uvEigh = np.linalg.eigh(A)
        self.evEigh = evEigh
        self.uvEigh = uvEigh
    
    def test_legendre_points(self):
        ''' Checks distribution points with the help of manual order '''
        fgk,fwk = read_fortranData()[2:4]
        gk,wk = quadraturePointsWeights(self.nc,self.quad,positiveHalf=False)
        np.testing.assert_allclose(fgk,gk[self.order],rtol=1e-5,atol=0)
        np.testing.assert_allclose(fwk,wk[self.order],rtol=1e-5,atol=0)

    def test_theta(self):
        ''' Checks angle for quadrature, theta '''
        ftheta= read_fortranData()[4]
        gk = quadraturePointsWeights(self.nc,self.quad,positiveHalf=False)[0]
        pi = np.pi
        theta = np.empty((self.nc))
        for k in range(self.nc):
            theta[k] = -(pi*0.5)*(gk[k]-1)
        np.testing.assert_allclose(ftheta,theta[self.order],rtol=1e-5,atol=0)
   
    def test_zne(self):
        ''' Checks quadrature points, zne '''
        fzne= read_fortranData()[5]
        r = abs(self.rmax-self.rmin)*0.5
        gk = quadraturePointsWeights(self.nc,self.quad,positiveHalf=False)[0]
        pi = np.pi
        zne = np.empty((self.nc),dtype=complex)
        for k in range(self.nc):
            theta = -(pi*0.5)*(gk[k]-1)
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta)+r*self.efactor*1.0j*math.sin(theta)
        np.testing.assert_allclose(fzne,zne[self.order],rtol=1e-5,atol=0)

    def test_Qe(self):
        ''' Checks linear solutions, Qe '''
        typeClass = self.guess[0].__class__
        r = abs(self.rmax-self.rmin)*0.5
        gk,wk = quadraturePointsWeights(self.nc,self.quad,positiveHalf=False)
        pi = np.pi
        zne = np.empty((self.nc),dtype=complex)
        n,m = len(self.guess),len(self.guess[0].array)
        Qe = np.empty((n,m),dtype=complex)
        for k in range(self.nc):
            theta = -(pi*0.5)*(gk[k]-1)
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta)+r*self.efactor*1.0j*math.sin(theta)
        
        zne = zne[self.order]    
        for k in range(self.nc):
            fQe = read_fortranData(k)[6]
            for im0 in range(len(self.guess)):
                Qe[im0] = typeClass.solve(self.mat,self.guess[im0],zne[k]).array
            np.testing.assert_allclose(Qe,fQe,rtol=1e-5,atol=0)

    def test_Q(self):
        ''' Checks integrated solutions, Q '''
        typeClass = self.guess[0].__class__
        r = abs(self.rmax-self.rmin)*0.5
        gk,wk = quadraturePointsWeights(self.nc,self.quad,positiveHalf=False)
        pi = np.pi
        theta = np.empty((self.nc))
        zne = np.empty((self.nc),dtype=complex)
        n,m = len(self.guess),len(self.guess[0].array)
        Q = [np.nan for it in range(n)]
        for k in range(self.nc):
            theta[k] = -(pi*0.5)*(gk[k]-1)
        
        theta = theta[self.order]
        wk = wk[self.order]
        for k in range(self.nc):
            fQ = read_fortranData(k)[7]
            zne[k] = ((self.rmin+self.rmax)*0.5)+ r*math.cos(theta[k])+r*self.efactor*1.0j*math.sin(theta[k])
            for im0 in range(len(self.guess)):
                Qquad_k = calculateQuadrature(self.mat,self.guess[im0],zne[k],r,theta[k],wk[k],self.efactor)
                Q = updateQ(Q,im0,Qquad_k,k)
            for im0 in range(len(self.guess)):
                np.testing.assert_allclose(Q[im0].array,fQ[im0],rtol=1e-5,atol=0)
    
        
if __name__ == '__main__':
    unittest.main()
