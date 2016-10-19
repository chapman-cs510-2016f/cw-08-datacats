#!/usr/bin/env python3

import cplanenp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Julia Plane Class
This module class extends the cplanenp class by creating the complex grid of numbers, then
transforming the plane using the class' Julia function, and turning it to the julia complex plane.
The class also has four functions to save the plane into a CSV and a JSON file. It can also build a plane by importing from a CSV or JSON file.
"""

class JuliaPlane(cplanenp.ComplexPlaneNP):
    """
    The class creates a plane of complex numbers and transforms it using the class' built in function, Julia.
    This transformation happens upon initialization and does not need to be called by the user by redefining the       initialization, with defaulted values at -2 to 2 in the x and y values, with 1000 points inbetween.
    The created plane will be a complex plane with those parameters, to transform them to the Julia set, the
    set_f() function needs to be called.

    This class also has functions to import and export the plane to a CSV and a JSON file.
        toCSV    :    saves the plane onto a CSV file.
        fromCSV  :    reads a CSV file and saves it to the plane variable
        toJSON   :    saves the plane onto a JSON file.
        fromJSON :    reads a JSON file and saves it to the plane variable
    The class has a function, show(), that will output 
    """
    def __init__(self, xmin=-2.0, xmax=2.0, xlen=1000, ymin=-2.0, ymax=2.0, ylen=1000, plane= [], f=np.vectorize(lambda x:x)):
        self.xmin = xmin
        self.xmax = xmax
        self.xlen = xlen
        self.ymin = ymin
        self.ymax = ymax
        self.ylen = ylen
        self.plane = plane
        self.f = f

        self.refresh()

    def show(self):
        """ Showcasing the Julia Set
        This function is used to show a transformed complex plane and displays it.
        Upon initialization, the module creates a complex set of numbers without applying a 
        Julia transformation. The user will need to use the funciton set_f() in order this method to
        properly display the Julia set.
        """
        # Setting our display dimensions and size
        plt.figure(figsize=(8,6), dpi=300)

        # Plotting the plane using imshow(), with a bicubic interpolation to help with smoothing
        # Also adding in a colorbar
        lm = plt.imshow(self.plane, extent=[-2,2,-2,2] , interpolation='bicubic', cmap='bone_r')
        plt.colorbar(shrink=.92)

        # setting the limits based on the min values, and length values.
        plt.xlim(-2.0, 2.0)
        plt.xticks(np.linspace(-2,2,5, endpoint=True))
        plt.ylim(-2.0, 2.0)
        plt.yticks(np.linspace(-2,2,5, endpoint=True))
        #plt.xticks(np.linspace(self.xmin,self.xmax,(self.xmax-self.xmin)+1, endpoint=True))
        #plt.yticks(np.linspace(self.xmin,self.xmax,(self.ymax-self.ymin)+1, endpoint=True))


        # Setting the axis so it lies in the middle of the image
        # ax= plt.gca() #getting current axis
        # ax.spines['right'].set_color('none')
        # ax.spines['top'].set_color('none')
        # ax.xaxis.set_ticks_position('bottom')
        # ax.spines['bottom'].set_position(('data',self.xlen/2))
        # ax.yaxis.set_ticks_position('left')
        # ax.spines['left'].set_position(('data',self.ylen/2))

        plt.show()

    def set_f(self, f=-1.037 + 0.17j):
        """This function applies a new function to be applied to the grid's values and recreates the grid
        This is a re-defined function of the cplanenp class' set_f function.
        Args:
            param1 (complex parameter)  : a complex parameter to transform the plane

        This function sets a new value to the class' function variable with the given parameter, calls
        the Julia class to transform the function and then recreates the grid by calling the refresh() function.
        """
        self.f = np.vectorize(self.julia(f))
        self.refresh()


    def toCSV(self, filename = "julia_plane.csv"):
        """ Saving dataframe to a CSV file
        This method will take in one argument
            param1 = name of file in string format
        and exports the plane to a csv file with the name given in the parameter.
        by default, the parameter will be named "julia_plane.csv" if no filename is passed in.
        """
        self.plane.to_csv(filename)

    def fromCSV(self, filename = "julia_plane.csv"):
        """ Reading from a CSV file
        This method will take in one argument
            param1 = name of file in string format
        and import the plane from a csv file with the name given in the parameter into a dataframe object.
        by default, the parameter will be named "julia_plane.csv" if no filename is passed in.
        """
        self.plane = pd.read_csv(filename)


    def toJSON(self, filename = "julia_plane.json"):
        """ Saving dataframe to a JSON file
        This method will take in one argument
            param1 = name of file in string format
        and exports the plane to a JSON file with the name given in the parameter.
        by default, the parameter will be named "julia_plane.json" if no filename is passed in.
        """
        self.plane.to_json(filename)

    def fromJSON(self, filename = "julia_plane.json"):
        """ Reading from a JSON file
        This method will take in one argument
            param1 = name of file in string format
        and import the plane from a JSON file with the name given in the parameter into a dataframe object.
        by default, the parameter will be named "julia_plane.json" if no filename is passed in.
        """
        self.plane = pd.read_json(filename)


