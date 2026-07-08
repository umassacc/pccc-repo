"""dfit.py 5/8/21 D. Candela
Module provides a simplified user interface to the nonlinear
least-squares fitting function scipy.optimize.curve_fit.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from scipy.optimize import curve_fit

class Dfit():
    """Object to carry out a least-squares fit using scipy.opitmize.curve_fit,
    and make the results available in various forms.
    
    The fit is carried out when a Dfit object is initialized, and then the
    fitted parameters and their two-sigma uncertainties are avaialable as the
    attributes ps, ups.
    
    The results of the fit can be printed or plotted by calling the methods
    printfit or plotfit.
    
    Parameters
    ----------
    f : function
        Model function.  Should take P+1 float arguments, which are the x
        value and the P parameters, and return a float.
    xs : array of float
        Indepdendent variable values for the data.
    ys : array of float
        Dependent variable values for the data.  Must have same length as xs.
    yerrs : (optional) array of float
        Standard deviations for the data points.  If supplied must have same
        length as xs.  If not supplied yerr=1.0 for every data point is used
        to compute chi-squared.
    datname : (optional) str
        Name of the data set.
    p0s : (optional) tuple, list, or array of float
        Initial values for the P parameters.  For variable parameters these
        are the initial values; for fixed parameters these are the fixed
        values. If not supplied initial value of 0.0 is used for all
        parameters.  If len(p0s)<P, initial value of 0.0 is used for the
        remaining parameters.
    varies : (optional) tuple, list, or array of bool
        If not supplied all P parameters are varied for the fit.  If supplied
        controls wich of the P parameters are varied in the fit; if
        len(varies)<P, the remaining parmeters not set by varies are fixed.
    units : (optional) tuple, list, or array of str
        Units string for each parameter, used when printing results.
    """
    def __init__(self,f,xs,ys,yerrs=None,datname=None,
                 p0s=None,varies=None,units=None):
        # Save the fitting function and the data.
        self.f = f
        self.xs = xs
        self.ys = ys
        self.yerrs = yerrs
        self.datname = datname
        self.nn = len(self.xs)
        # Get the names of the fitting function and its parameters.
        self.fname = f.__code__.co_name
        self.pp = f.__code__.co_argcount-1   # number of model parameters
        self.fparams = f.__code__.co_varnames[1:self.pp+1] # their names
        # Get varies augmented to actual number of parameters pp.
        if varies is None:
            self.ppvaries = [True]*self.pp
        else:
            self.ppvaries = [False]*self.pp
            for ip,vip in enumerate(varies):
                if ip<self.pp and varies[ip]:
                    self.ppvaries[ip] = True
        # Get initial values augmented to actual number of parameters pp.
        self.ppp0s = [0.]*self.pp
        if p0s:
            for ip,p0ip in enumerate(p0s):
                if ip<self.pp:
                    self.ppp0s[ip] = float(p0s[ip])
                    # (if not float printfit chokes on p0s)
        # Get list of units strings augmented to actual number of params pp.
        self.ppunits = ['']*self.pp
        if units:
            for ip,uip in enumerate(units):
                if ip<self.pp:
                    self.ppunits[ip] = units[ip]
        # List full-parameter index for each variable-param index.
        self.ips = [ip for ip in range(self.pp) if self.ppvaries[ip]]
        self.cf_pp = len(self.ips)   # number of variable parameters
        if self.cf_pp:  # if there are any variable params..
            # ..do the fit.  Start by extracting the initial values (guesses)
            # for the variable parameters:
            cf_p0s = [self.ppp0s[ip] for ip in self.ips]
            # Next define fitting function for curve_fit.
            def cf_f(x,*cf_ps):
                # Get full param list (includes fixed params).
                ps = self.fullps(cf_ps)
                # Return user's fitting function with full param list.
                return f(x,*ps)
            # Do the fit using curve_fit.
            cf_ps,self.cc = curve_fit(cf_f,xs,ys,p0=cf_p0s,sigma=yerrs)
            # Save full-param list of results (includes fixed params).
            self.ps = self.fullps(cf_ps)
        self.ups =[0.]*self.pp  # fixed params have uncertainties of zero
        # If there were variable parameters, compute uncertainties.
        if self.cf_pp:
            # Compute 2-sigma uncertainties as 2 times square root of
            # variances, which are diagonal elements of covariance matrix
            # self.cc returnced by curve_fit.
            cf_ups = 2*sqrt(np.diag(self.cc))
            for cf_ip in range(self.cf_pp):
                self.ups[self.ips[cf_ip]] = cf_ups[cf_ip]

    def fullps(self,cf_ps):
        """Helper for __init__: Returns full (length self.pp) parameter list
        which includes fixed parameter values and supplied list cf_ps of
        variable parameter values.
        """
        # Start with initial values, which include the fixed values.
        ps = self.ppp0s.copy()
        # Substitute supplied values of variable params.
        for cf_ip in range(self.cf_pp):
            ps[self.ips[cf_ip]] = cf_ps[cf_ip]
        return ps
        
    def printfit(self):
        """Print out the fit results."""
        if self.datname:
            dset = f'{self.datname}, '
        else:
            dset = ''
        print(f'DATA SET: {dset}{self.nn} points')
        print(f'FITTING FUNCTION: {self.fname},'
              f' {self.cf_pp} variable parameters')
        # Compute and print chi-squared and reduced chi-squared.
        errs = not self.yerrs is None
        yfits = self.f(self.xs,*self.ps)
        yys = (yfits-self.ys)**2
        if errs:
            yys = ((yfits-self.ys)/self.yerrs)**2
        else:
            yys = (yfits-self.ys)**2
        chisq = sum(yys)
        print(f'Chi-squared = {chisq:.4}',end='')
        dof = self.nn-self.cf_pp    # degrees of freedom
        if dof>0:
            print(f'  Reduced chi-squared = {chisq/dof:.4}',end='')
            if not errs:
                print(' (no yerrs supplied)',end='')
        print()
        # Print parameter values and uncertainties.
        print('Parameters and 2-sigma parameter uncertainties:')
        for ip in range(self.pp):
            print(f'{self.fparams[ip]:>10} = {self.ps[ip]:<11.5}',end='')
            if self.ppvaries[ip]:
                print(f' +/- {self.ups[ip]:<10.5} ',end='')
            else:
                print(f'     {"(fixed)":<10} ',end='')
            print(self.ppunits[ip])

    def plotfit(self,xlabel=None,ylabel=None,title=None):
        """Plot the fit results and the data together.

        Parameters
        ----------
        xlabel, ylabel,title : (optional) str
            If supplied, used to label plot axes and to title plot
        """
        # Make arrays xmod, ymod of the model with 500 points covering a
        # slightly larger x range than the data.
        datmin = min(self.xs)
        datmax = max(self.xs)
        xmin = datmin - 0.05*(datmax-datmin)
        xmax = datmax + 0.05*(datmax-datmin)
        xmod = np.linspace(xmin,xmax,500)
        ymod = self.f(xmod,*self.ps)
        # Make title string for plot unless one is supplied.
        if title:
            tstr = title
        else:
            tstr = f'{self.cf_pp}-param fit of {self.fname}'
            if self.datname:
                tstr += f' to data set: {self.datname}'
        # Plot the data.
        if not self.yerrs is None:
            plt.errorbar(self.xs,self.ys,self.yerrs,fmt='bo',capsize=4)
        else:
            plt.plot(self.xs,self.ys,'bo')
        # Plot the model.
        plt.plot(xmod,ymod,'r')
        # Add titles, labels, and display.
        plt.title(tstr)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.show()
        
""" ****************** End of module dfity.py *********************** """