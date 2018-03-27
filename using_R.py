#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:41:19 2018

@author: juanerolon
"""

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages

from rpy2.robjects.packages import importr
# import R's "base" package
base = importr('base')

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=2) # select the first mirror in the list

package_name = 'epitools'

utils.install_packages(package_name)

if False:
    help_doc = utils.help('help')
    print(help_doc)
    

#More complex strings are R expressions of arbitrary complexity, 
#or even sequences of expressions (snippets of R code). 
#Their evaluation is performed in what is known to R users as 
#the Global Environment, that is the place one starts at when in the R console.
# Whenever the R code creates variables, those variables are “located” 
#in that Global Environment by default.
#For example, the string below returns the value 18.85

    
ro.r('''
        # create a function `f`
        f <- function(r, verbose=FALSE) {
            if (verbose) {
                cat("I am calling f().\n")
            }
            2 * pi * r
        }
        # call the function `f` with argument value 3
        f(3)
        ''')
    
#That string is a snippet of R code (complete with comments) 
#that first creates an R function, then binds it to the symbol f (in R), 
#finally calls that function f. The results of the call 
#(what the R function f is returns) is returned to Python.
#Since that function f is now present in the R Global Environment, 
#it can be accessed with the __getitem__ mechanism outlined above:

    
r_f = ro.globalenv['f']
print(r_f.r_repr())

#The function r_f is callable, and can be used like a regular Python function

res = r_f(3)
print(res)





