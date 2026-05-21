import util
from datetime import datetime
from util_funcs import find_nearest
import numpy as np
from numpyVector import NumpyVector
from ttnsVector import TTNSVector

# ****************************************************************************
def convert(arr,eShift=0.0,unit='au'):
    """Convert an ndarray (energy or matrix) with an eShift and unit conversion."""

    arrShifted = None
    if unit == 'au':
        arrShifted = arr - eShift
    else:
        arrShifted = util.au2unit(arr,unit)-eShift
    return arrShifted

# ****************************************************************************
#                   Print modules for LANCZOS
# ****************************************************************************
class LanczosPrintUtils:
    """Print helper for file headers, footers, and iteration output."""
    def __init__(self,guessVector,sigma,L,maxit,eConv,checkFitTol, 
            writeOut,eShift,convertUnit,pick,status,
            outFileName=None, summaryFileName=None):

        if outFileName is None:
            outFileName = "iterations_lanczos.out"
        if summaryFileName is None:
            summaryFileName = "summary_lanczos.out"

        self.typeClass = guessVector.__class__
        self.options = guessVector.options
        self.sigma = sigma
        self.L = L
        self.maxit = maxit
        self.eConv = eConv
        self.checkFitTol = checkFitTol
        self.writeOut = writeOut
        self.eShift = eShift
        self.convertUnit = convertUnit
        self.pick = pick
        self.status = status
        if self.writeOut:
            self.outfile = open(outFileName,"w")
            self.sumfile = open(summaryFileName,"w")
        else:
            self.outfile = None
            self.sumfile = None

    def __del__(self):
        if self.outfile is not None:
            self.outfile.close()
        if self.sumfile is not None:
            self.sumfile.close()

    def fileHeader(self):
        """Print the header with all input information."""
        if not self.writeOut:
            return

        # .....................  get typeClass & options .....................
        if "linearSystemArgs" in self.options:
            optLinear = self.options["linearSystemArgs"]
        if "stateFittingArgs" in self.options:
            optFitting = self.options["stateFittingArgs"]
        if "linearSystemArgs" in self.options:
            optOrtho = self.options["linearSystemArgs"]

        # ..........................  Open files .............................
        outfile = self.outfile
        sumfile = self.sumfile

        sumfile.write("startingPoint"+"\n") # data extractor
        # ..........................  timeStamp ..............................
        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        dt_string = "\t\t"+dateTime+"\t\t\n"
        lines = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+\
                "*"*70+"\n\n"

        # ..........................  general info ..........................
        nBlock = self.status["nBlock"]
        lines += f"# Inexact Lanczos with {nBlock} guess vectors"+"\n\n"

        formatStyle = "{:12} {:>14} :: {:20}"
        target = convert(self.sigma,self.eShift,self.convertUnit)
        lines += formatStyle.format("target",f"{target:.2f}",\
                "target excitation")+"\n"
        lines += formatStyle.format("L",self.L,"Krylov space")+"\n"
        lines += formatStyle.format("maxit",self.maxit,\
                "Maximum Lanczos iterations")+"\n"
        lines += formatStyle.format("econv",f"{self.eConv:.03g}",\
                "Eigenvalue convergence")+"\n"
        lines += formatStyle.format("checkFitTol",self.checkFitTol,"Checkfit tolerance")+"\n"
        pickname = str(self.pick).split(" ")[1]
        lines += "{:10} {:>20}".format("pick",pickname)+"\n"

        # ..........................  sweep info numpyVector.................
        if self.typeClass is NumpyVector:
            solver = optLinear["linearSolver"]
            linearTol = optLinear["linear_tol"]
            nsweepLinear = optLinear["linearIter"]

            lines += formatStyle.format("lsweep",nsweepLinear,\
                    "Number of sweeps: Linear solver")+"\n"
            lines += formatStyle.format("solver",solver,"Linear solver")+"\n"
            lines += formatStyle.format("ltol",linearTol,\
                    "Tolerance: Linear solver")+"\n"
    # ..........................  sweep info ttnsVector......................
        elif self.typeClass is TTNSVector:
            solver = optLinear["iterativeLinearSystemOptions"].solver
            siteLinearTol = optLinear["iterativeLinearSystemOptions"].tol
            maxIter = optLinear["iterativeLinearSystemOptions"].maxIter
            globalLinearTol = optLinear["convTol"]
            nsweep = optLinear["nSweep"]
            adaptLinear = optLinear["bondDimensionAdaptions"]

            lines += formatStyle.format("solver",solver,"Linear solver")+"\n"
            lines += formatStyle.format("ltol1",siteLinearTol,\
                    "Site tolerance: Linear solver")+"\n"
            lines += formatStyle.format("maxIter",maxIter,\
                    "Iterative solver maximum iterations")+"\n"
            lines += formatStyle.format("lsweep",nsweep,\
                    "Number of DMRG sweeps: Linear solver")+"\n"
            lines += formatStyle.format("ltol2",globalLinearTol,\
                    "Global tolerance: Linear solver")+"\n"
            lines += formatStyle.format("maxD",adaptLinear[0].maxD if adaptLinear is not None else -1,\
                    "Maximum bond dimension: Linear solver")+"\n"

            fittingTol = optFitting["convTol"]
            nsweepFitting = optFitting["nSweep"]
            adaptFitting = optFitting["bondDimensionAdaptions"]

            lines += formatStyle.format("ftol",fittingTol,"Fitting tolerance")+"\n"
            lines += formatStyle.format("fsweep",nsweepFitting,\
                    "Number of sweeps: fitting")+"\n"
            lines += formatStyle.format("maxD",adaptFitting[0].maxD if adaptFitting is not None else -1,\
                    "Maximum bond dimension: Fitting")+"\n"

        # ..........................  Space for phase calculations ..........
        lines += formatStyle.format("Phase",self.status["phase"],\
                "Stage of phase calculation")+"\n\n"

        # ..........................  write and print info ..................
        outfile.write(lines)
        sumfile.write(lines)
        print(lines)

        # ..........................  data description in plot file ..........
        lines = "{:>4} {:>6} {:>6} {:>12}".format("it","i","nCum","target")
        for iBlock in range(nBlock):
            lines += "{:>18}".format("EvalueBlock"+str(iBlock+1))
        lines += "{:>16} {:>16}".format("residual","time(seconds)"+"\n")
        
        sumfile.write(lines)
    
        outfile.flush()
        sumfile.flush()
# ****************************************************************************

    def fileFooter(self):
        """Print the footer with the job-complete message."""
        if not self.writeOut:
            return
        sumfile = self.sumfile
        outfile = self.outfile
        sumfile.write("endingPoint"+"\n") # for data extractor code

    # ..........................  timeStamp ..........................
        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        dt_string = "\t\t"+dateTime+"\t\t\n"
        line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+dt_string+\
                "*"*70+"\n\n"
        outfile.write(line+"\n")
        sumfile.write(line+"\n")
        print(line)

        outfile.flush()
        sumfile.flush()
        return
# ****************************************************************************
    
    def writeFile(self,label,*args):
        """A single print function for overlap, Hamiltonian matrix,
        iteration details, bond dimensions, summary outputs and 
        final eigenvalues."""

        if not self.writeOut:
            return
        sumfile = self.sumfile
        outfile = self.outfile

    # ........................ OVERLAP MATRIX ........................
        if label == "overlap":
            Smat = args[0]
            condSmat = np.linalg.cond(Smat)
            outfile.write("\n"+f"overlap condition number {condSmat:5.3e}")
            outfile.write("\nOVERLAP MATRIX\n")
            outfile.write(f"{Smat}")
            outfile.write("\n\n")
    # ........................ HAMILTONIAN MATRIX ......................
        elif label == "hamiltonian":
            outfile.write("HAMILTONIAN MATRIX\n")
            outfile.write(args[1]+"\n")
            hmat = convert(args[0],self.eShift,self.convertUnit)
            outfile.write(f"{hmat}")
            outfile.write("\n\n")
    # ........................ EIGENVALUES ..............................
        elif label == "eigenvalues":
            outfile.write("Eigenvalues\n")
            evalues = convert(args[0],self.eShift,self.convertUnit)
            outfile.write(f"{evalues}")
            outfile.write("\n")
    # ...................... ITERATION INFO ..............................
        elif label == "iteration":
            line = "\n\n"+"."*20+"\tInfo per iteration\t"+"."*20+"\n"
            line += "Lanczos iteration: "+str(args[0]["outerIter"])
            line += "\tKrylov iteration: "+str(args[0]["innerIter"])
            line += "\tCumulative Krylov iteration: "+str(args[0]["cumIter"])+"\n"
            outfile.write(line)
            if self.typeClass is TTNSVector:print(line) # for sweepOutput
    # ...................... MAXIMUM BOND DIMENSION ......................
        elif label == "KSmaxD":
            line = "Maximum bond dimensions of Krylov vectors"
            KSmaxD = args[0]["KSmaxD"]
            line += f"{KSmaxD}"+"\n\n"
            outfile.write(line)
        elif label == "fitmaxD":
            line = "Maximum bond dimensions of fitted vectors"
            fitmaxD = args[0]["fitmaxD"]
            line += f"{fitmaxD}"+"\n\n"
            outfile.write(line)
    # ........................FINAL RESULTS ..............................
        elif label == "results":
            # same as 'eigenvalues', with ev_nearest and final message
            lines = "\n\n" +"-"*20+"\tFINAL RESULTS\t"+"-"*20+"\n"
            energies = convert(args[0],self.eShift,self.convertUnit)
            lines += "All subspace eigenvalues:\n"
            lines += f"{energies}"+"\n"
            target = convert(self.sigma,self.eShift,self.convertUnit)
            ev_nearest = find_nearest(energies,target)[1]
            lines += f"Target, Lanczos (nearest) {target}, {ev_nearest}\n"
            outfile.write(lines)
    # ....................... SUMMARY FILE ..............................
        elif label == "summary":
            status = args[1]
            nBlock = status["nBlock"]
            it = status["outerIter"]
            i = status["innerIter"]
            nCum = status["cumIter"]
            runTime = status["runTime"]

            target = convert(self.sigma,self.eShift,self.convertUnit)
            evalue = convert(args[0],unit=self.convertUnit)
            excitation = convert(evalue,eShift=self.eShift)
            residual = status["residual"]

            lines = "{:>4} {:>6} {:>6} {:>12}".format(it,i,nCum,\
                    f"{target:5.2f}")
            
            for iBlock in range(nBlock):
                lines += "{:>18}".format(f"{excitation[iBlock]:.10f}")
            
            lines += "{:>16} {:>16}".format(f"{residual:5.4e}",f"{runTime:.2f}"\
                    +"\n")
            sumfile.write(lines)
        
        outfile.flush()
        sumfile.flush()
        return

# ****************************************************************************
#                   Print modules for FEAST
# ****************************************************************************
class FeastPrintUtils:
    """Print helper for file headers, footers, and iteration output."""
    def __init__(self,guessVector,n_quad,quad,rmin,rmax,eConv,maxit,writeOut,
            eShift,convertUnit,status,outFileName=None, summaryFileName=None):
        
        if outFileName is None:
            outFileName = "iterations_feast.out"
        if summaryFileName is None:
            summaryFileName = "summary_feast.out"

        self.typeClass = guessVector.__class__
        self.subspace = len(guessVector)
        self.options = guessVector[0].options
        self.n_quad = n_quad
        self.quad = quad
        self.rmin = rmin
        self.rmax = rmax
        self.eConv = eConv
        self.maxit = maxit
        self.writeOut = writeOut
        self.eShift = eShift
        self.convertUnit = convertUnit
        self.status = status
        if self.writeOut:
            self.outfile = open(outFileName,"w")
            self.sumfile = open(summaryFileName,"w")
        else:
            self.outfile = None
            self.sumfile = None

    def __del__(self):
        if self.outfile is not None:
            self.outfile.close()
        if self.sumfile is not None:
            self.sumfile.close()

    def fileHeader(self):
        """Print the header with all input information."""
        if not self.writeOut:
            return
         
        # .....................  get typeClass & options .....................
        if "linearSystemArgs" in self.options:
            optLinear = self.options["linearSystemArgs"]
        if "stateFittingArgs" in self.options:
            optFitting = self.options["stateFittingArgs"]
        if "linearSystemArgs" in self.options:
            optOrtho = self.options["linearSystemArgs"]

        sumfile = self.sumfile
        outfile = self.outfile
        sumfile.write("startingPoint"+"\n") # data extractor
        # ..........................  timeStamp ..............................
        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        dt_string = "\t\t"+dateTime+"\t\t\n"
        lines = "*"*70 + "\n\t\tStarting computation\t\t\n"+dt_string+\
                "*"*70+"\n\n"

        # ..........................  general info ..........................
        formatStyle = "{:12} {:>14} :: {:20}"
        lines += formatStyle.format("m0",self.subspace,\
                "Subspace dimension")+"\n"
        lines += formatStyle.format("n_quad",self.n_quad,"Number of quadrature points")\
                +"\n"
        lines += formatStyle.format("quad",self.quad,\
                "Quadrature distribution")+"\n"
        minTarget = convert(self.rmin,self.eShift,self.convertUnit)
        lines += formatStyle.format("emin",minTarget,\
                "Minimum target excitation energy")+"\n"
        maxTarget = convert(self.rmax,self.eShift,self.convertUnit)
        lines += formatStyle.format("emax",maxTarget,\
                "Maximum target excitation energy")+"\n"
        lines += formatStyle.format("econv",f"{self.eConv:.03g}",\
                "Eigenvalue convergence")+"\n"
        lines += formatStyle.format("maxit",self.maxit,\
                "Maximum FEAST iterations")+"\n"
        lines += formatStyle.format("eShift",self.eShift,"shift energy")+"\n"
        lines += formatStyle.format("convertUnit",self.convertUnit,"convertUnit")+"\n"

        # ..........................  sweep info numpyVector.................
        if self.typeClass is NumpyVector:
            solver = optLinear["linearSolver"]
            linearTol = optLinear["linear_tol"]
            nsweepLinear = optLinear["linearIter"]

            lines += formatStyle.format("lsweep",nsweepLinear,\
                    "Number of sweeps: Linear solver")+"\n"
            lines += formatStyle.format("solver",solver,"Linear solver")+"\n"
            lines += formatStyle.format("ltol",linearTol,\
                    "Tolerance: Linear solver")+"\n"
    # ..........................  sweep info ttnsVector......................
        elif self.typeClass is TTNSVector:
            solver = optLinear["iterativeLinearSystemOptions"].solver
            siteLinearTol = optLinear["iterativeLinearSystemOptions"].tol
            maxIter = optLinear["iterativeLinearSystemOptions"].maxIter
            globalLinearTol = optLinear["convTol"]
            nsweep = optLinear["nSweep"]
            adaptLinear = optLinear["bondDimensionAdaptions"]

            lines += formatStyle.format("solver",solver,"Linear solver")+"\n"
            lines += formatStyle.format("ltol1",siteLinearTol,\
                    "Site tolerance: Linear solver")+"\n"
            lines += formatStyle.format("maxIter",maxIter,\
                    "Iterative solver maximum iterations")+"\n"
            lines += formatStyle.format("lsweep",nsweep,\
                    "Number of DMRG sweeps: Linear solver")+"\n"
            lines += formatStyle.format("ltol2",globalLinearTol,\
                    "Global tolerance: Linear solver")+"\n"
            lines += formatStyle.format("maxD",adaptLinear[0].maxD,\
                    "Maximum bond dimension: Linear solver")+"\n"

            fittingTol = optFitting["convTol"]
            nsweepFitting = optFitting["nSweep"]
            adaptFitting = optFitting["bondDimensionAdaptions"]

            lines += formatStyle.format("ftol",fittingTol,"Fitting tolerance")+"\n"
            lines += formatStyle.format("fsweep",nsweepFitting,\
                    "Number of sweeps: fitting")+"\n"
            lines += formatStyle.format("maxD",adaptFitting[0].maxD,\
                    "Maximum bond dimension: Fitting")+"\n"

        # ..........................  Space for phase calculations ..........
        lines += formatStyle.format("Phase",self.status["phase"],\
                "Stage of phase calculation")+"\n\n"

        # ..........................  write and print info ..................
        outfile.write(lines)
        sumfile.write(lines)
        print(lines)

        # ..........................  data description in plot file ..........
        lines = "{:>4} {:>6}".format("it","quad")
        
        for iSubspace in range(self.subspace):
            lines += "{:>16}".format("Evalue"+str(iSubspace+1))
        lines += "{:>16} {:>16}".format("residual","time(seconds)"+"\n")
        
        sumfile.write(lines)
        outfile.flush()
        sumfile.flush()
        return
# ****************************************************************************

    def fileFooter(self):
        """Print the footer with the job-complete message."""
        if not self.writeOut:
            return
        sumfile = self.sumfile
        outfile = self.outfile
        # ..........................  Open files ..........................
        sumfile.write("endingPoint"+"\n") # for data extractor code

    # ..........................  timeStamp ..........................
        dateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        dt_string = "\t\t"+dateTime+"\t\t\n"
        line = "\n"+"*"*70 + "\n\t\tEnd of computation\t\t\n"+dt_string+\
                "*"*70+"\n\n"
        outfile.write(line+"\n")
        sumfile.write(line+"\n")
        print(line)
        outfile.flush()
        sumfile.flush()
        return
# ****************************************************************************
    
    def writeFile(self,label,*args):
        """A single print function for overlap, Hamiltonian matrix,
        iteration details, summary output and final eigenvalues."""
        if not self.writeOut:
            return
        sumfile = self.sumfile
        outfile = self.outfile
    # ........................ OVERLAP MATRIX ........................
        if label == "overlap":
            outfile.write("OVERLAP MATRIX\n")
            outfile.write(f"{args[0]}")
            outfile.write("\n\n")
    # ........................ HAMILTONIAN MATRIX ......................
        elif label == "hamiltonian":
            outfile.write("HAMILTONIAN MATRIX\n")
            hmat = convert(args[0],self.eShift,self.convertUnit)
            outfile.write(f"{hmat}")
            outfile.write("\n\n")
    # ........................ EIGENVALUES ..............................
        elif label == "eigenvalues":
            outfile.write("Eigenvalues\n")
            evalues = convert(args[0],self.eShift,self.convertUnit)
            outfile.write(f"{evalues}")
            outfile.write("\n")
    # ...................... ITERATION INFO ..............................
        elif label == "iteration":
            line = "\n\n"+"."*20+"\tInfo per iteration\t"+"."*20+"\n"
            line += "FEAST iteration: "+str(args[0]["outerIter"])+"\n"
            outfile.write(line)
            print(line) # for sweepOutput
    # ....................... SUMMARY FILE ..............................
        elif label == "summary":
            status = args[2]
            it = status["outerIter"]
            quad = status["quadrature"]
            runTime = status["runTime"]

            evalue = convert(args[0],unit=self.convertUnit)
            excitation = convert(evalue,eShift=self.eShift)
            residual = args[1]

            lines = "{:>4} {:>6}".format(it,quad)
           
           # NOTE: The len(excitation) != subspace case is missing for FEAST.
           # After orthogonalization, the number of eigenvalues can differ
           # from the subspace size, so the summary-file header needs updating.
            for iExcitation in range(len(excitation)):
                lines += "{:>16}".format(f"{excitation[iExcitation]:.08f}")
            
            lines += "{:>16} {:>16}".format(f"{residual:5.4e}",f"{runTime:.2f}"\
                    +"\n")
            sumfile.write(lines)
            
        outfile.flush()
        sumfile.flush()
        return
# **************************************************************************************
