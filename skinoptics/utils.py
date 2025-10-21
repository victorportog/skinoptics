'''
| SkinOptics
| Copyright (C) 2024-2025 Victor Lima

    | This program is free software: you can redistribute it and/or modify
    | it under the terms of the GNU General Public License as published by
    | the Free Software Foundation, either version 3 of the License, or
    | (at your option) any later version.

    | This program is distributed in the hope that it will be useful,
    | but WITHOUT ANY WARRANTY; without even the implied warranty of
    | MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    | GNU General Public License for more details.

    | You should have received a copy of the GNU General Public License
    | along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
| Victor Lima
| victor.lima\@ufscar.br
| victorportog.github.io

| Release date:
| October 2024
| Last modification:
| October 2024

| References:

| [WSS13] Wyman, Sloan & Shirley 2013.
| Simple Analytic Approximations to the CIE XYZ Color Matching Functions
| https://jcgt.org/published/0002/02/01/
'''

import numpy as np

def dict_from_arrays(array_keys, array_values):
    r'''
    Construct a dictionary from two arrays.
    
    :param array_keys: array with the dictionary keys
    :type array_keys: np.ndarray
    
    :param array_values: array with the dictionary values
    :type array_values: np.ndarray
    
    :return: - **dict** (*dictionary*) – dictionary constructed with the two arrays
    '''
    
    return {array_keys[i]: array_values[i] for i in range(len(array_keys))}
    
def circle(r, xc, yc, theta_i = 0., theta_f = 360., n_points = 1000):
    r'''
    The points that compose a circle.
    
    :math:`\left \{ \begin{matrix}
    x = r \cos \theta + x_c\\
    y = r \sin \theta + y_c
    \end{matrix} \right.`
        
    :param r: radius
    :type r: float
    
    :param xc: center x-coordinate
    :type xc: float
    
    :param yc: center y-coordinate
    :type yc: float
    
    :param theta_i: initial angle [degrees] (default to 0.)
    :type theta_i: float
    
    :param theta_f: final angle [degrees] (default to 360.)
    :type theta_f: float
          
    :param n_points: number of points (default to 1000)
    :type n_points: int
    
    :return: - **x** (*np.ndarray*) – x-coordinantes of a circle
    
       - **y** (*np.ndarray*) – y-coordinantes of a circle
    '''
    
    theta = np.linspace(theta_i*np.pi/180., theta_f*np.pi/180., n_points)
    x = r*np.cos(theta) + xc
    y = r*np.sin(theta) + yc
    
    return x, y
    
def linear(x, a, b):
    r'''
    The linear function.
    
    :math:`f(x) = ax + b`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: slope
    :type a: float

    :param b: y-intercept
    :type b: float

    :return: - **f** (*float or np.ndarray*) – evaluated linear function 
    '''
    
    return a*x + b
    
def quadratic(x, a, b, c):
    r'''
    The quadratic function.
    
    :math:`f(x) = ax^2 + bx + c`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :return: - **f** (*float or np.ndarray*) – evaluated quadratic function
    '''
    
    return a*np.power(x, 2, dtype = 'float64') + b*x + c

def cubic(x, a, b, c, d):
    r'''
    The cubic function.
    
    :math:`f(x) = ax^3 + bx^2 + cx + d`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :param d: function constant
    :type d: float

    :return: - **f** (*float or np.ndarray*) – evaluated cubic function
    '''
    
    return a*np.power(x, 3, dtype = 'float64') + b*np.power(x, 2, dtype = 'float64') + c*x + d

def exp_decay(x, a, b, c):
    r'''
    The exponential decay function.
    
    :math:`f(x) = a \mbox{ exp}(-|b|x) + c`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :return: - **f** (*float or np.ndarray*) – evaluated exponential decay function
    '''
    
    return a*np.exp(-np.abs(b)*x) + c

def biexp_decay(x, a, b, c, d, e):
    r'''
    The biexponential decay function.
    
    :math:`f(x) = a\mbox{ exp}(-|b|x) + c \mbox{ exp}(-|d|x) + e`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :param d: function constant
    :type d: float

    :param e: function constant
    :type e: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated biexponential decay function
    '''
        
    return a*np.exp(-np.abs(b)*x) + c*np.exp(-np.abs(d)*x)  + e

def exp_decay_inc_form(x, a, b, c):
    r'''
    The increasing form of the exponential decay function.
    
    :math:`f(x) = a \mbox{ [}1 - \mbox{exp}(-|b|x)] + c`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated exponential decay (increasing form) function
    '''
    
    return a*(1 - np.exp(-np.abs(b)*x)) + c

def natural_log(x, a, b, c):
    r'''
    The natural logarithm function.
    
    :math:`f(x) = a \mbox{ ln}(x + b) + c`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated natural logarithm function
    '''
        
    return a*np.log(x + b) + c

def gaussian(x, a, b, c):
    r'''
    The Gaussian function.
    
    :math:`f(x) = a \mbox{ exp}\left[-\frac{1}{2}\frac{(x - b)^2}{c^2}\right]`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated Gaussian function
    '''
    
    return a*np.exp(-1./2.*(x - b)**2/c**2)

def heaviside(x):
    r'''
    The Heaviside step function.
    
    :math:`H(x) =  
    \left \{ \begin{matrix}
    0, & \mbox{if } g < 0 \\
    \frac{1}{2}, & \mbox{if } g = 0 \\
    1, & \mbox{if } g > 0 \\
    \end{matrix} \right.`
    
    :param x: function variable
    :type x: float or np.ndarray
    
    :return: - **H** (*float or np.ndarray*) – evaluated Heaviside step function
    '''
    
    if isinstance(x, np.ndarray) == True:
        H = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] < 0.:
                H[i] = 0.
            elif x[i] == 0.:
                H[i] = 1./2.
            else:
                H[i] = 1.        
    elif isinstance(x, (int, float)) == True:
        if x < 0.:
            H = 0.
        elif x == 0.:
            H = 1./2.
        else:
            H = 1.
    else:
        msg = 'x must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return H
    
def mod_gaussian_Wyman(x, a, b, c, d):
    r'''
    | The modified Gaussian function needed to calculate some analytical functions 
    | from Wyman, Sloan & Shirley 2013 [WSS13] (see function :meth:`skinoptics.colors.cmfs`).
    
    :math:`f(x) = a \mbox{ exp}\left\{-d \mbox{ } \left[\mbox{ ln}\left(\frac{x - b}{c}\right)\right]^2\right\}`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :param d: function constant
    :type d: float

    :return: - **f** (*float or np.ndarray*) – evaluated function
    '''
        
    return a*np.exp(-d*(np.log((x - b)/c))**2)

def selector_function_Wyman(x, y, z): 
    r'''
    | The selector function needed to calculate some analytical functions 
    | from Wyman, Sloan & Shirley 2013 [WSS13] (see function :meth:`skinoptics.colors.cmfs`).
    
    :math:`S(x,y,z) = y \mbox{ } (1 - H(x)) + z \mbox{ } H(x)`
    
    in which :math:`H(x)` is the Heaviside step function :meth:`skinoptics.utils.heaviside`.
    
    :param x: function variable
    :type x: float or np.ndarray

    :param y: function variable
    :type y: float or np.ndarray

    :param z: function variable
    :type z: float or np.ndarray
    
    :return: - **S** (*float or np.ndarray*) – evaluated selector function
    '''
    
    return y*(1 - heaviside(x)) + z*heaviside(x)

def piecewise_gaussian_Wyman(x, a, b, c, d):
    r'''
    | The piecewise Gaussian function needed to calculate some analytical functions 
    | from Wyman, Sloan & Shirley 2013 [WSS13] (see function :meth:`skinoptics.colors.cmfs`).
    
    :math:`f(x) = a \mbox{ exp}\left\{-\frac{1}{2}[(x - b)\mbox{ }S(x - b \mbox{, }c \mbox{, }d)]^2\right\}`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :param d: function constant
    :type d: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated function
    '''
    
    return a*np.exp(-1./2.*((x - b)*selector_function_Wyman(x - b, c, d))**2)

def hyperbolic_cossine(x, a, b, c, d):
    r'''
    The hyperbolic cossine function (with four parameters).
    
    :math:`f(x) = a \mbox{ cosh}\left[\frac{(x - b)}{c}\right] + d`
    
    :param x: function variable
    :type x: float or np.ndarray

    :param a: function constant
    :type a: float

    :param b: function constant
    :type b: float

    :param c: function constant
    :type c: float

    :param d: function constant
    :type d: float
    
    :return: - **f** (*float or np.ndarray*) – evaluated hyperbolic cossine function
    '''
    
    return a*np.cosh((x - b)/c) + d
