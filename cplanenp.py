#!/usr/bin/env python3

import abscplane as ab
import numpy as np
import pandas as pd
import numba as nb 

"""Complex Plane Creation
This module imports from the class abscplane, and inherits the class' object
and variables.
This module expands on the class by creating a complex number plane,
creating a 2 dimensional grid of complex numbers in the form of (x + y * 1j)
where 1j is the imaginary untit, defined in mathematics as i; the square root of -1.
The exect specifications of the grid is left to the user.
The class as attributes of the following:
        xmax (float) : maximum horizontal axis value
        xmin (float) : minimum horizontal axis value
        xlen (int)   : number of horizontal points
        ymax (float) : maximum vertical axis value
        ymin (float) : minimum vertical axis value
        ylen (int)   : number of vertical points
        plane        : stored complex plane implementation
        f    (func)  : function displayed in the plane


This module also has 3 aditional functions that are used to transform the 2d grid as needed.:
    refresh : redraws the plane creation
    zoom    : resets graph's x and y values to "zoom in/out", and redraws the graph with new perameters
    set_f   : resets the transformation function for the graph, then redraws the graph.
This module also has a function Julia
"""

class ComplexPlaneNP(ab.AbsComplexPlane):
    """Complex Plane Class: Sets the initial values for the object, as well as creates the plane.

         The creation takes in the following arugments, and has default values if no value is passed through:

            param1 (float) : minimum horizontal, X, axis value. Default value of -5
            param2 (float) : maximum horizontal, X, axis value. Default value of 5
            param3 (int)   : number of horizontal points between the x-min and x-max values. Default value of 10.
            param4 (float) : minimum vertical, Y, axis value. Default value of -5
            param5 (float) : maximum vertical, Y, axis value. Default value of 5
            param6 (int)   : number of vertical points between the y-min and y-max values. Default value of 10
            param7         : stored complex plane implementation. Defaulted as a black list, []
            param8 (func)  : function displayed in the plane. Default function as the identity function. x:x

        The initialization will also call the class function, refresh() to create the plane with the given
        arugments.
    """

    def __init__(self, xmin=-5.0, xmax=5.0, xlen=10, ymin=-5.0, ymax=5.0, ylen=10, plane= [], f=np.vectorize(lambda x:x)):
        self.xmin = xmin
        self.xmax = xmax
        self.xlen = xlen
        self.ymin = ymin
        self.ymax = ymax
        self.ylen = ylen
        self.plane = plane
        self.f = f

        # calling refresh() to build the plane
        self.refresh()

    def refresh(self):
        """
        For every point (x + y*1j) in the plane, replace
        the point with the value self.f(x + y*1j).
        This function will take the set values of xmin, xmax, xlen, ymin, ymax, ylen to
        create the plane.
        It will also take the set value of f and apply the function upon creation of the plane,
        using numpy and pandas to create the plane and give it labels.
        """
        # Plane creationg using numpy by first using linspace to get all the x and y values between their respective min and max and the points inbetween by their length values.
        # Then uing meshgrid to combine all the points together and using the formula x + y*1j to vectorize the entire grid.
        a = np.linspace(self.xmin, self.xmax, self.xlen+1)
        b = np.linspace(self.ymin, self.ymax, self.ylen+1)
        y, x = np.meshgrid(b,a)
        z = x + y*1j
        z = self.f(z)

        # Creating a list of each x and y value to be used in our pandas implementation for column and row names.
        x_val = []
        y_val = []
        inc_x = (self.xmax - self.xmin) / (self.xlen)
        inc_y = (self.ymax - self.ymin) / (self.ylen)
        for i in range(self.xlen + 1):
            x_val.append(self.xmin + (i * inc_x) )
        for j in range(self.ylen + 1):
            y_val.append(str(self.ymin + (j * inc_y)) + "*j")

        # Using pandas Dataframe to add x and y value labels to the grid, then transposing the grid so the values are ordered to resemble the number plane.
        f = pd.DataFrame(z, x_val, y_val)
        self.plane = f.T

        # For testing purposes, print out the plane to check if everything works and the values are correct.
        # print(self.plane)


    def zoom(self, xmin, xmax, xlen, ymin, ymax, ylen):
        """This function zooms into the graph, given by the parameters
        Args:
            param1 (float) : The new value for the minimum horizontal axis
            param2 (float) : The new value for the maximum horizontal axis
            param3 (int)   : The new value for the horizontal points between the x-max and x-min values.
            param4 (float) : The new value for the maximum horizontal axis
            param5 (float) : The new value for the minimum horizontal axis
            param6 (int)   : mThe new value for the vertical points between the y-max and y-min values.

        This function takes in a user input of xmin, xmax, xlen, ymin, ymax, ylen and resets
        the class values to the parameters.
        The function will then 'zoom in' by recreating the graph, given the newly defined values
        by calling the refresh() function.
        """
        self.xmin = xmin
        self.xmax = xmax
        self.xlen = xlen
        self.ymin = ymin
        self.ymax = ymax
        self.ylen = ylen

        self.refresh()

    def set_f(self, f):
        """This function applies a new function to be applied to the grid's values and recreates the grid
        Args:
            param1 (complex parameter)  : a complex parameter to transform the plane

        This function sets a new value to the class' function variable with the given parameter and then
        recreates the grid by calling the refresh() function.
        """
        self.f = f
        self.refresh()
        
   
    def julia(self, c, max=100):
        """Julia
        This module has a method called Julia, which takes in the following arguments:
            c           : a complex parameter
            max (int)   : an optional positive integer, defaulted at 100
        and returns the function f.
        The function f takes in the argument
            z           : a complex parameter
        and returns an integer value, n.
        Upon being called, the function f will do the following operation z = z**2 + c, and will keep track of how many
        iterations of the operation will it take before the absolute value of z exceeds the value of 2, stores that value in the variable n
        then returns n.
        The method will return 0 if n exceeds the max value.
        """
        # error message to check that max is a positive integer value
        if max < 1:
            print("Please input a positive integer value for max")
            return 0
        @nb.vectorize([nb.int32(nb.complex128)])
        # function f
        def f(z):
            # checks to make sure that the value of z is already greater than 2
            if abs(z) > 2:
                return 1

            # setting our initial variables
            n = 0
            test = True

            # While loop that will iterate as many times as needed until the value of z exceeds 2, or if the value of n exceeds max
            while test:
                z = z**2 + c
                n += 1
                if n == max:
                    test = False
                    n = 0
                elif abs(z) > 2:
                    test = False
            return n
        return f
        
