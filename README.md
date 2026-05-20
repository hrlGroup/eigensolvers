# Description
Targeted eigensolvers to find an eigenstate (or a set of eigenstates) in a specified range.
The main implementation is an inexact Lanczos approach[1,3]. 
Another, preliminary and not fully tested implementation is the FEAST approach[2].
These implementations use a general interface that supports both NumPy arrays (`NumpyVector` wrapper) and tensor network states (`TTNSVector` wrapper). The tensor-network implementation currently depends on in-house code, which will be released separately.
Note: **This is work in progress**. 

# Theoretical background
Suppose **H** is a Hermitian matrix whose eigenvalues and eigenvectors are to be calculated. 
In many cases, the full spectrum is not required. Assume we want a few eigenvectors near the eigenvalue $\sigma$.
In this case, instead of solving the actual matrix, **H**, it is often better to solve the transformed form, **F(H)**. 
This transformation is often called a spectral transform, and it is chosen to increase the separation between the desired eigenvalues near $\sigma$ and the rest of the spectrum. 
With this larger separation, the target part of the spectrum is easier to converge and fewer iterations are needed compared to working with **H** directly.
One straightforward way to increase the separation near $\sigma$ is the shift-and-invert approach, which uses $F(H)=(\sigma -H)^{-1}$.
F(**H**)**v** is evaluated approximately by solving the linear system ($\sigma$**I**-**H**)**w** = **v** iteratively. 
The eigenvalue problem in the **w** basis is then solved to obtain eigenvalues and corresponding eigenvectors, resulting in the inexact Lanczos approach.
In the FEAST approach, the eigenvalues are computed through contour integration.
Rather than targeting a specific eigenvalue, FEAST finds eigenvalues within a specified range.

# Prerequisites (recommended version)
1. Python (3.10.14)
2. SciPy (1.10.1)
3. NumPy (1.26.4)

# Unittest
Before running NumPyVector tests, run the following unit tests from the `unittests` folder:
1. test_lanczos.py 
2. test_lanczosBlock.py
3. test_lanczosLINDEP.py

# Working examples
An example (`driver_numpyVector.py`) is available in the `examples` folder.

# Input arguments
Inexact Lanczos eigensolver
1. H  		: diagonalizable input matrix or linear operator
2. v0 		: eigenvector guess
     		  Can be a list of `AbstractVectors`.
     		  Then, block Lanczos is performed (Krylov space on each of the guesses).
     		  Note that the guess vectors should be orthogonal.
3. sigma 		: eigenvalue estimate
4. L  		: Krylov space dimension
5. maxit 		: maximum number of Lanczos iterations
6. eConv 		: relative eigenvalue convergence tolerance
7. checkFitTol 
(optional) 	: tolerance for checking fitted vectors
8. Hsolve
 (optional) 	: Like H, but only used to generate Lanczos vectors.
                  `H` is still used to diagonalize the Hamiltonian matrix.
9. writeOut
(optional) 	: whether to write output files
             	  Default: write both `iterations_lanczos.out` and `summary_lanczos.out`.
10. eShift 
(optional) 	: shift value for eigenvalues and Hmat elements
11. convertUnit 
(optional) 	: unit conversion for eigenvalues and Hmat elements
12. pick 
(optional) 	: pick function for eigenstate
                  Default is `get_pick_function_close_to_sigma`.
13. status 
(optional) 	: Additional information dictionary
                  See the `_getStatus` docstring for details.
14. outFileName 
(optional)	: output file name
15. summaryFileName
(optional)	: summary file name

# Output files
After a successful test or example-driver run, the following output files are written:
1. `iterations_lanczos.out`: detailed information at each cumulative iteration, including parameter details, the overlap matrix, the Hamiltonian matrix before and after diagonalization, and eigenvalues.
2. `summary_lanczos.out`: summary information at each cumulative iteration, along with parameter details in the file header.

# Contributors
1. Dr. Madhumita Rano
2. Prof. Henrik R. Larsson (https://github.com/hrlGroup)

# References
1. Shi-Wei Huang and Tucker Carrington Jr., “A new iterative method for calculating energy levels and
wave functions”, The Journal of Chemical Physics 112.20 (2000), pp. 8765–8771.
2. Eric Polizzi., “Density-matrix-based algorithm for solving eigenvalue problems”, Physical Review B
79.11 (2009), p. 115112.
3. Madhumita Rano and Henrik R. Larsson, Computing excited eigenstates using inexact Lanczos methods and tree tensor network states, J. Chem. Phys., 163 (2025), 164110, https://doi.org/10.1063/5.0301263
    arXiv preprint arXiv:2506.22574
