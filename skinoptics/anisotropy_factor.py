'''
| SkinOptics
| Copyright (C) 2024 Victor Lima

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
| victorporto\@ifsc.usp.br
| victor.lima\@ufscar.br

| Release Date:
| October 2024
| Last Modification:
| October 2025

| References:

| [HG41] Henyey & Greenstein 1941.
| Diffuse radiation in the Galaxy.
| https://doi.org/10.1086/144246

| [RM80] Reynolds & McCormick 1980.
| Approximate two-parameter phase function for light scattering.
| https://doi.org/10.1364/JOSA.70.001206

| [Bv84] Bruls & van der Leun 1984.
| Forward scattering properties of human epidermal layers.
| https://doi.org/10.1111/j.1751-1097.1984.tb04581.x

| [JAP87] Jacques, Alter & Prahl 1987.
| Angular Dependence of HeNe Laser Light Scattering by Human Dermis.
| https://omlc.org/~prahl/pubs/pdfx/jacques87b.pdf

| [Y*87] Yoon, Welch, Motamedi & van Gemert 1987.
| Development and Application of Three-Dimensional Light Distribution Model for Laser Irradiated Tissue.
| https://doi.org/10.1109/JQE.1987.1073224

| [v*89] van Gemert, Jacques, Sterenborg & Star 1989.
| Skin Optics.
| https://doi.org/10.1109/10.42108

| [CS92] Cornette & Shanks 1992.
| Physically reasonable analytic expression for the single-scattering phase function
| https://doi.org/10.1364/AO.31.003152

| [WJ92] Wang & Jacques 1992.
| Monte Carlo Modeling of Light Transport in Multi-layered Tissues in Standard C.
| https://omlc.org/software/mc/mcml/MCman.pdf

| [D03] Draine 2003.
| Scattering by Interstellar Dust Grains. I. Optical and Ultraviolet.
| https://doi.org/10.1086/379118

| [F11] Frisvad 2011.
| Importance sampling the Rayleigh phase function.
| https://doi.org/10.1364/JOSAA.28.002436

| [B*14] Bosschaart, Edelman, Aalders, van Leeuwen & Faber 2014.
| A literature review and novel theoretical approach on the optical properties of whole blood.
| https://doi.org/10.1007/s10103-013-1446-7

| [BCK22] Baes, Camps & Kapoor 2022.
| A new analytical scattering phase function for interstellar dust.
| https://doi.org/10.1051/0004-6361/202142437

| [JM23] Jacques & McCormick 2023.
| Two-term scattering phase function for photon transport to model subdiffuse reflectance
| in superficial tissues.
| https://doi.org/10.1364/BOE.476461
'''

import numpy as np
from scipy.interpolate import interp1d

from skinoptics.utils import *
from skinoptics.dataframes import *

def ptheta_R(theta):
    r'''
    | The Rayleigh scattering phase function.
    | For details please check Frisvad 2011 [F11].
    
    :math:`p_{R}(\theta) = \frac{3}{8}(1 + \cos^2\theta)`

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    return (3/8)*(1 + np.cos(theta*np.pi/180)**2)

def ptheta_HG(theta, g):
    r'''
    | The Henyey-Greenstein scattering phase function.
    | For details please check Henyey & Greenstein 1941 [HG41].
    
    :math:`p_{HG}(\theta, g) = \frac{1}{2}\frac{1 - g^2}{(1 + g^2 - 2g \cos \theta )^{3/2}}`

    In this particular model :math:`g` is the anisotropy factor.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
        
    return (1/2)*(1 - g**2)/(1 + g**2 - 2*g*np.cos(theta*np.pi/180))**(3/2)

def ptheta_HGIT(theta, g, gamma):
    r'''
    | The Henyey-Greenstein scattering phase function with an isotropic term.
    | For details please check Jacques, Alter & Prahl 1987 [JAP87] and Yoon et al. [Y*87].
    
    :math:`p_{HGIT}(\theta, g, \gamma) = \frac{1}{2}\gamma + (1-\gamma) \mbox{ } p_{HG}(\theta, g)`

    In this model :math:`g` is NOT the anisotropy factor.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :param gamma: relative weight of the isotropic term component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
    if gamma < 0 or gamma > 1:
        msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
        raise Exception(msg)
        
    return (1/2)*gamma + (1 - gamma)*ptheta_HG(theta = theta, g = g)

def ptheta_TTHG(theta, g1, g2, gamma):
    r'''
    | The two-term Henyey-Greenstein scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`p_{TTHG}(\theta, g_1, g_2, \gamma) = \gamma \mbox{ } p_{HG}(\theta, g_1) + (1 - \gamma) \mbox{ } p_{HG}(\theta, g_2)`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g1 < 0 or g1 > 1:
        msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
        raise Exception(msg)
    if g2 < -1 or g2 > 0:
        msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
        raise Exception(msg)
    if gamma < 0 or gamma > 1:
        msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
        raise Exception(msg)
    
    return gamma*ptheta_HG(theta = theta, g = g1) + (1 - gamma)*ptheta_HG(theta = theta, g = g2)

def ptheta_RM(theta, g, alpha):
    r'''
    | The Reynolds-McCormick scattering phase function.
    | For details please check Reynolds & McCormick 1980 [RM80] and Jacques & McCormick 2023 [JM23].
    
    :math:`p_{RM}(\theta, g, \alpha) = 2 \frac{\alpha g}{(1 + g)^{2\alpha} - (1 - g)^{2\alpha}}\frac{(1 - g^2)^{2\alpha}}{(1 + g^2 - 2g\cos\theta)^{\alpha + 1}}`
    

    | For :math:`\alpha = 1/2` it reduces to the Henyey-Greenstein scattering phase function.
    | For :math:`\alpha = 1` it reduces to the Ultraspherical-2 scattering phase function.

    | In this model :math:`g` is the anisotropy factor only when :math:`\alpha = 1/2`.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float 
    
    :param alpha: parameter :math:`\alpha` [-] (must be greater than -0.5)
    :type alpha: float 
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
    if alpha <= -1/2:
        msg = 'The input alpha = {} is not greater than -1/2.'.format(alpha)
        raise Exception(msg)
        
    return 2*(alpha*g)/((1 + g)**(2*alpha) - (1 - g)**(2*alpha))*((1 - g**2)**(2*alpha))/((1 + g**2 - 2*g*np.cos(theta*np.pi/180))**(alpha + 1))

def ptheta_TTRM(theta, g1, g2, alpha1, alpha2, gamma):
    r'''
    | The two-term Reynolds-McCormick scattering phase function.
    | For details please check  Reynolds & McCormick 1980 [RM80] and Jacques & McCormick 2023 [JM23].
    
    :math:`p_{TTRM}(\theta, g_1, g_2, \alpha_1, \alpha_2, \gamma) =  \gamma \mbox{ } p_{RM}(\theta, g_1, \alpha_1) + (1 - \gamma) \mbox{ } p_{RM}(\theta, g_2, \alpha_2)`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param alpha1: parameter :math:`\alpha_1` [-] (must be greater than -1/2)
    :type alpha1: float
    
    :param alpha2: parameter :math:`\alpha_2` [-] (must be greater than -1/2)
    :type alpha2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g1 < 0 or g1 > 1:
        msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
        raise Exception(msg)
    if g2 < -1 or g2 > 0:
        msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
        raise Exception(msg)
    if alpha1 <= -1/2:
        msg = 'The input alpha1 = {} is not greater than -1/2.'.format(alpha1)
        raise Exception(msg)
    if alpha2 <= -1/2:
        msg = 'The input alpha2 = {} is not greater than -1/2.'.format(alpha2)
        raise Exception(msg)
    if gamma < 0 or gamma > 1:
        msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
        raise Exception(msg)
    
    return gamma*ptheta_RM(theta = theta, g = g1, alpha = alpha1) + (1 - gamma)*ptheta_RM(theta = theta, g = g2, alpha = alpha2)

def ptheta_CS(theta, g):
    r'''
    | The Cornette-Shanks scattering phase function.
    | For details please check Cornette & Shanks 1992 [CS92].
    
    :math:`p_{CS}(\theta, g) = \frac{3}{2}\frac{1 + \cos^2\theta}{2 + g^2} \mbox{ } p_{HG}(\theta, g)`

    | In this model :math:`g` is NOT the anisotropy factor.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
    
    return (3/2)*(1 + np.cos(theta*np.pi/180.)**2)/(2 + g**2)*ptheta_HG(theta = theta, g = g)

def ptheta_D(theta, g, alpha):
    r'''
    | The Draine scattering phase function.
    | For details please check Draine 2003 [D03].
    
    :math:`p_{D}(\theta, g, \alpha) = 3\frac{1 + \alpha \cos^2\theta}{3 + \alpha (1 + 2g^2)} \mbox{ } p_{HG}(\theta, g)`
    
    | For :math:`\alpha = 1` and :math:`g = 0` it reduces to the Rayleigh scattering phase function.
    | For :math:`\alpha = 0` it reduces to the Henyey-Greenstein scattering phase function.
    | For :math:`\alpha = 1` it reduces to the Cornette-Shanks scattering phase function

    | In this model :math:`g` is NOT the anisotropy factor.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float

    :param alpha: parameter :math:`\alpha` [-]
    :type alpha: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
    
    return 3*(1 + alpha*np.cos(theta*np.pi/180.)**2)/(3 + alpha*(1 + 2*g**2))*ptheta_HG(theta = theta, g = g)

def ptheta_U2(theta, g):
    r'''
    | The Ultraspherical-2 scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`p_{U2}(\theta, g) = \frac{1}{2}\frac{(1 - g^2)^2}{(1 + g^2 - 2g \cos \theta)^2}`

    | In this model :math:`g` is NOT the anisotropy factor.

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
        
    return (1/2)*(1 - g**2)**2/(1 + g**2 - 2*g*np.cos(theta*np.pi/180.))**2

def ptheta_TTU2(theta, g1, g2, gamma):
    r'''
    | The two-term Ultraspherical-2 scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`p_{TTU2}(\theta, g_1, g_2, \gamma) = \gamma \mbox{ } p_{U2}(\theta, g_1) + (1 - \gamma) \mbox{ } p_{U2}(\theta, g_2)`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak

    :param theta: scattering angle [degrees]
    :type theta: float or np.ndarray
    
    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **ptheta** (*float or np.ndarray*) – scattering phase function [-]
    '''
    
    if g1 < 0 or g1 > 1:
        msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
        raise Exception(msg)
    if g2 < -1 or g2 > 0:
        msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
        raise Exception(msg)
    if gamma < 0 or gamma > 1:
        msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
        raise Exception(msg)
    
    return gamma*ptheta_U2(theta = theta, g = g1) + (1 - gamma)*ptheta_U2(theta = theta, g = g2)

def theta_R_from_RND(n_RND = int(1E6)):
    r'''
    | The scattering angle distribution as a function of a set of random numbers uniformly
    | distributed over the interval [0, 1), assuming the Rayleigh scattering phase function.
    | For details please check section 3.B from Frisvad 2011 [F11].
    
    | :math:`\theta_{R} = \mbox{arccos}(\sqrt[3]{u + v} + \sqrt[3]{u - v})`
    | with
    | :math:`u = -2(2 \xi - 1)`
    | :math:`v = \sqrt{4(2 \xi - 1)^2 + 1}`
    | in which :math:`\xi` is a random number in the interval [0, 1)

    :param n_RND: number of random numbers [-] (default to int(1E6))
    :type n_RND: int
    
    :return: - **theta** (*np.ndarray*) – scattering angle [degrees]
    '''
    
    xi = np.random.rand(n_RND)
    two_times_xi_minus_one = 2*xi - 1
    term1 = -2*two_times_xi_minus_one
    term2 = np.sqrt(4*two_times_xi_minus_one**2 + 1)
    
    return np.arccos(np.cbrt(term1 + term2) + np.cbrt(term1 - term2))*180/np.pi


def theta_HG_from_RND(g, n_RND = int(1E6)):
    r'''
    | The scattering angle distribution as a function of the anisotropy factor and a set of random
    | numbers uniformly distributed over the interval [0, 1), assuming the Henyey-Greenstein
    | scattering phase function.
    | For details please check section 3.5 from Wang & Jacques 1992 [WJ92].
    
    :math:`\theta_{HG} =  
    \left \{ \begin{matrix}
    \mbox{arccos}(2 \xi - 1) , & \mbox{if } g = 0 \\
    \mbox{arccos}\left\{\frac{1}{2g} \left[1 + g^2 - \left(\frac{1 - g^2}{1 - g + 2g \xi}\right)^2\right]\right\}, & \mbox{if } g \ne 0
    \end{matrix} \right.` 
   
    in which :math:`\xi` is a random number in the interval [0, 1)

    In this particular model :math:`g` is the anisotropy factor.

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :param n_RND: number of random numbers [-] (default to int(1E6))
    :type n_RND: int
    
    :return: - **theta** (*np.ndarray*) – scattering angle [degrees]
    '''
    
    if g < -1 or g > 1:
        msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
        raise Exception(msg)
    
    xi = np.random.rand(n_RND)
    if g == 0:    
        theta_HG = np.arccos(2*xi - 1)*180./np.pi
    else:
        theta_HG = np.arccos(1/(2*g)*(1 + g**2 - ((1 - g**2)/(1 - g + 2*g*xi))**2))*180./np.pi
        
    return theta_HG

def theta_U2_from_RND(g, n_RND = int(1E6)):
    r'''
    | The scattering angle distribution as a function of the g parameter and a set of random
    | numbers uniformly distributed over the interval [0, 1), assuming the Ultraspherical-2
    | scattering phase function.
    | For details please check section 4.4.2 from Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`\theta_{U2} = arccos\left[\frac{(1 + g)^2 - 2 \xi (1 + g^2)}{(1 + g)^2 - 4g \xi}\right]`

    in which :math:`\xi` is a random number in the interval [0, 1)

    | In this model :math:`g` is NOT the anisotropy factor.

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :param n_RND: number of random numbers [-] (default to int(1E6))
    :type n_RND: int
    
    :return: - **theta** (*np.ndarray*) – scattering angle [degrees]
    '''
    
    xi = np.random.rand(n_RND)
    
    return np.arccos(((1 + g)**2 - 2*xi*(1 + g**2))/((1 + g)**2 - 4*g*xi))*180./np.pi

def costheta_HGIT(g, gamma):
    r'''
    | The anisotropy factor as a function of the parameters g and gamma, assuming the
    | Henyey-Greenstein scattering phase function with an isotropic term.
    | For details please check Jacques, Alter & Prahl 1987 [JAP87] and Yoon et al. [Y*87].
    
    :math:`\langle \cos\theta \rangle_{HGIT}(g, \gamma) = (1 - \gamma) \mbox{ } g`

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :param gamma: fraction of isotropic term contribution [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g, np.ndarray) == True:
        if np.any(g < -1) or np.any(g > 1):
            msg = 'At least one element in the input g is out of the range [-1, 1].'
            raise Exception(msg)
    elif isinstance(g, (int, float)) == True:
        if g < -1 or g > 1:
            msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
            raise Exception(msg)
    else:
        msg = 'The input g must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(gamma, np.ndarray) == True:
        if np.any(gamma < 0) or np.any(gamma > 1):
            msg = 'At least one element in the input gamma is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(gamma, (int, float)) == True:
        if gamma < 0 or gamma > 1:
            msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
            raise Exception(msg)
    else:
        msg = 'The input gamma must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return (1 - gamma)*g

def costheta_TTHG(g1, g2, gamma):
    r'''
    | The anisotropy factor as a function of the parameters g1, g2 and gamma, assuming the
    | two-term Henyey-Greenstein scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`\langle \cos\theta \rangle_{TTHG}(g_1, g_2, \gamma) = \gamma \mbox{ } g_1 + (1 - \gamma) \mbox { } g_2`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak

    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g1, np.ndarray) == True:
        if np.any(g1 < 0) or np.any(g1 > 1):
            msg = 'At least one element in the input g1 is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(g1, (int, float)) == True:
        if g1 < 0 or g1 > 1:
            msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
            raise Exception(msg)
    else:
        msg = 'The input g1 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(g2, np.ndarray) == True:
        if np.any(g2 < -1) or np.any(g2 > 0):
            msg = 'At least one element in the input g2 is out of the range [-1, 0].'
            raise Exception(msg)
    elif isinstance(g2, (int, float)) == True:
        if g2 < -1 or g2 > 0:
            msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
            raise Exception(msg)
    else:
        msg = 'The input g2 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(gamma, np.ndarray) == True:
        if np.any(gamma < 0) or np.any(gamma > 1):
            msg = 'At least one element in the input gamma is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(gamma, (int, float)) == True:
        if gamma < 0 or gamma > 1:
            msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
            raise Exception(msg)
    else:
        msg = 'The input gamma must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return gamma*g1 + (1 - gamma)*g2

def costheta_RM(g, alpha):
    r'''
    | The anisotropy factor as a function of the parameters g and alpha, assuming the
    | Reynolds-McCormick scattering phase function.
    | For details please check Reynolds & McCormick 1980 [RM80] and Jacques & McCormick 2023 [JM23].
    
    | :math:`\langle \cos\theta \rangle_{RM}(g, \alpha) = \frac{2 \alpha g L - (1+g^2)}{2g(\alpha - 1)}`
    | with
    | :math:`L = \frac{(1+g)^{2\alpha} + (1-g)^{2\alpha}}{(1+g)^{2\alpha} - (1-g)^{2\alpha}}`

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float

    :param alpha: parameter :math:`\alpha` [-] (must be greater than -0.5)
    :float alpha: float or np.ndarray
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g, np.ndarray) == True:
        if np.any(g < -1) or np.any(g > 1):
            msg = 'At least one element in the input g is out of the range [-1, 1].'
            raise Exception(msg)
    elif isinstance(g, (int, float)) == True:
        if g < -1 or g > 1:
            msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
            raise Exception(msg)
    else:
        msg = 'The input g must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(alpha, np.ndarray) == True:
        if np.any(alpha <= -1/2):
            msg = 'At least one element in the input alpha is not greater than -1/2.'
            raise Exception(msg)
        costheta = np.zeros((len(alpha), 5))
        for i in range(len(alpha)):
            if i == 1:
                if isinstance(g, np.ndarray) == True:
                    costheta[i] = costheta_U2(g[i])
                else:
                    costheta[i] = costheta_U2(g)
            else:
                if isinstance(g, np.ndarray) == True:
                    one_plus_g = 1 + g[i]
                    one_minus_g = 1 - g[i]
                    two_times_alpha = 2*alpha[i]
                    L = (one_plus_g**two_times_alpha + one_minus_g**two_times_alpha)/(one_plus_g**two_times_alpha - one_minus_g**two_times_alpha)
                    costheta[i] = (2*alpha[i]*g[i]*L - (1 + g[i]**2))/(2*g[i]*(alpha[i] - 1))
                else:
                    one_plus_g = 1 + g
                    one_minus_g = 1 - g
                    two_times_alpha = 2*alpha
                    L = (one_plus_g**two_times_alpha + one_minus_g**two_times_alpha)/(one_plus_g**two_times_alpha - one_minus_g**two_times_alpha)
                    costheta[i] = (2*alpha[i]*g*L - (1 + g**2))/(2*g*(alpha[i] - 1))        
    elif isinstance(alpha, (int, float)) == True:
        if alpha <= -1/2:
            msg = 'The input alpha = {} is not greater than -1/2.'.format(alpha)
            raise Exception(msg)
        if alpha == 1:
            costheta = costheta_U2(g)
        else:
            one_plus_g = 1 + g
            one_minus_g = 1 - g
            two_times_alpha = 2*alpha
            L = (one_plus_g**two_times_alpha + one_minus_g**two_times_alpha)/(one_plus_g**two_times_alpha - one_minus_g**two_times_alpha)
            costheta = (2*alpha*g*L - (1 + g**2))/(2*g*(alpha - 1))    
    else:
        msg = 'The input alpha must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return costheta

def costheta_TTRM(g1, g2, alpha1, alpha2, gamma):
    r'''
    | The anisotropy factor as a function of the parameters g1, g2, alpha1, alpha2 and gamma,
    | assuming the two-term Reynolds-McCormick scattering phase function.
    | For details please check Reynolds & McCormick 1980 [RM80] and Jacques & McCormick 2023 [JM23].
    
    :math:`\langle \cos\theta \rangle_{TTRM}(g_1, g_2, \alpha_1, \alpha_2, \gamma) = \gamma \mbox{ } \langle \cos\theta \rangle_{RM}(g_1, \alpha_1) + (1 - \gamma) \mbox{ } \langle \cos\theta \rangle_{RM}(g_2, \alpha_2)`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak

    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param alpha1: parameter :math:`\alpha_1` [-] (must be greater than -0.5)
    :type alpha1: float
    
    :param alpha2: parameter :math:`\alpha_2` [-] (must be greater than -0.5)
    :type alpha2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g1, np.ndarray) == True:
        if np.any(g1 < 0) or np.any(g1 > 1):
            msg = 'At least one element in the input g1 is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(g1, (int, float)) == True:
        if g1 < 0 or g1 > 1:
            msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
            raise Exception(msg)
    else:
        msg = 'The input g1 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(g2, np.ndarray) == True:
        if np.any(g2 < -1) or np.any(g2 > 0):
            msg = 'At least one element in the input g2 is out of the range [-1, 0].'
            raise Exception(msg)
    elif isinstance(g2, (int, float)) == True:
        if g2 < -1 or g2 > 0:
            msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
            raise Exception(msg)
    else:
        msg = 'The input g2 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(alpha1, np.ndarray) == True:
        if np.any(alpha1 <= 1/2):
            msg = 'At least one element in the input alpha1 is not greater than 1/2.'
            raise Exception(msg)
    elif isinstance(alpha1, (int, float)) == True:
        if alpha1 <= 1/2:
            msg = 'The input alpha1 = {} is not greater than 1/2.'.format(alpha1)
            raise Exception(msg)
    else:
        msg = 'The input alpha1 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(alpha2, np.ndarray) == True:
        if np.any(alpha2 <= 1/2):
            msg = 'At least one element in the input alpha2 is not greater than 1/2.'
            raise Exception(msg)
    elif isinstance(alpha2, (int, float)) == True:
        if alpha2 <= 1/2:
            msg = 'The input alpha2 = {} is not greater than 1/2.'.format(alpha2)
            raise Exception(msg)
    else:
        msg = 'The input alpha2 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(gamma, np.ndarray) == True:
        if np.any(gamma < 0) or np.any(gamma > 1):
            msg = 'At least one element in the input gamma is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(gamma, (int, float)) == True:
        if gamma < 0 or gamma > 1:
            msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
            raise Exception(msg)
    else:
        msg = 'The input gamma must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return gamma*costheta_RM(g = g1, alpha = alpha1) + (1 - gamma)*costheta_RM(g = g2, alpha = alpha2)

def costheta_CS(g):
    r'''
    | The anisotropy factor as a function of the parameter g, assuming the Cornette-Shanks
    | scattering phase function.
    | For details please check Cornette & Shanks 1992 [CS92].
    
    :math:`\langle \cos\theta \rangle_{CS}(g) = \frac{3(4 + g^2)}{5(2 + g^2)} \mbox{ } g`

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g, np.ndarray) == True:
        if np.any(g < -1) or np.any(g > 1):
            msg = 'At least one element in the input g is out of the range [-1, 1].'
            raise Exception(msg)
    elif isinstance(g, (int, float)) == True:
        if g < -1 or g > 1:
            msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
            raise Exception(msg)
    else:
        msg = 'The input g must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return g*(3*(4 + g**2))/(5*(2 + g**2))

def costheta_D(g, alpha):
    r'''
    | The anisotropy factor as a function of the parameters g and alpha, assuming the Draine
    | scattering phase function.
    | For details please check Draine 2003 [D03].
    
    :math:`\langle \cos\theta \rangle_{D}(g, \alpha) = \frac{1 + \alpha(3 + 2g^2)/5}{1 + \alpha(1 + 2g^2)/3} \mbox{ } g`

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float

    :param alpha: parameter :math:`\alpha` [-]
    :float alpha: float or np.ndarray
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g, np.ndarray) == True:
        if np.any(g < -1) or np.any(g > 1):
            msg = 'At least one element in the input g is out of the range [-1, 1].'
            raise Exception(msg)
    elif isinstance(g, (int, float)) == True:
        if g < -1 or g > 1:
            msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
            raise Exception(msg)
    else:
        msg = 'The input g must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return g*(1 + alpha*(3 + 2*g**2)/5)/(1 + alpha*(1 + 2*g**2)/3)

def costheta_U2(g):
    r'''
    | The anisotropy factor as a function of the parameter g, assuming the Ultraspherical-2
    | scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`\langle \cos\theta \rangle_{U2}(g) = \frac{1+g^2}{2g} + \left(\frac{1-g^2}{2g}\right)^2 \mbox{ ln} \left(\frac{1-g}{1+g}\right)`

    :param g: parameter :math:`g` [-] (must be in the range [-1, 1])
    :type g: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g, np.ndarray) == True:
        if np.any(g < -1) or np.any(g > 1):
            msg = 'At least one element in the input g is out of the range [-1, 1].'
            raise Exception(msg)
    elif isinstance(g, (int, float)) == True:
        if g < -1 or g > 1:
            msg = 'The input g = {} is out of the range [-1, 1].'.format(g)
            raise Exception(msg)
    else:
        msg = 'The input g must be int, float or np.ndarray.'
        raise Exception(msg)
        
    return (1 + g**2)/(2*g) + ((1 - g**2)/(2*g))**2*np.log((1 - g)/(1 + g))

def costheta_TTU2(g1, g2, gamma):
    r'''
    | The anisotropy factor as a function of the parameters g1, g2 and gamma, assuming the
    | two-term Ultraspherical-2 scattering phase function.
    | For details please check Baes, Camps & Kapoor 2022 [BCK22].
    
    :math:`\langle \cos\theta \rangle_{TTU2}(g_1, g_2, \gamma) = \gamma \mbox{ } \langle \cos\theta \rangle_{U2}(g_1) + (1 - \gamma) \mbox{ } \langle \cos\theta \rangle_{U2}(g_2)`

    :math:`g_1` characterizes the shape and the strength of the forward scattering peak

    :math:`g_2` characterizes the shape and the strength of the backward scattering peak
    
    :param g1: parameter :math:`g_1` [-] (must be in the range [0, 1])
    :type g1: float
    
    :param g2: parameter :math:`g_2` [-] (must be in the range [-1, 0])
    :type g2: float
    
    :param gamma: relative weight of the forward scattering component [-] (must be in the range [0, 1])
    :type gamma: float
    
    :return: - **costheta** (*np.ndarray*) – anisotropy factor [-]
    '''
    
    if isinstance(g1, np.ndarray) == True:
        if np.any(g1 < 0) or np.any(g1 > 1):
            msg = 'At least one element in the input g1 is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(g1, (int, float)) == True:
        if g1 < 0 or g1 > 1:
            msg = 'The input g1 = {} is out of the range [0, 1].'.format(g1)
            raise Exception(msg)
    else:
        msg = 'The input g1 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(g2, np.ndarray) == True:
        if np.any(g2 < -1) or np.any(g2 > 0):
            msg = 'At least one element in the input g2 is out of the range [-1, 0].'
            raise Exception(msg)
    elif isinstance(g2, (int, float)) == True:
        if g2 < -1 or g2 > 0:
            msg = 'The input g2 = {} is out of the range [-1, 0].'.format(g2)
            raise Exception(msg)
    else:
        msg = 'The input g2 must be int, float or np.ndarray.'
        raise Exception(msg)
        
    if isinstance(gamma, np.ndarray) == True:
        if np.any(gamma < 0) or np.any(gamma > 1):
            msg = 'At least one element in the input gamma is out of the range [0, 1].'
            raise Exception(msg)
    elif isinstance(gamma, (int, float)) == True:
        if gamma < 0 or gamma > 1:
            msg = 'The input gamma = {} is out of the range [0, 1].'.format(gamma)
            raise Exception(msg)
    else:
        msg = 'The input gamma must be int, float or np.ndarray.'
        raise Exception(msg)
    
    return gamma*costheta_U2(g = g1) + (1 - gamma)*costheta_U2(g = g2)

def g_vanGemert(lambda0):
    r'''
    | The anisotropy factor of human EPIDERMIS or DERMIS as a function of wavelength.
    | van Gemert et al. 1989 [v*89]'s fit for experimental data from Bruls & van der Leun 1984
    | [Bv84] (epidermis, 302, 365, 436 and 546 nm) and Jacques, Alter & Prahl 1987 [JAP87]
    | (dermis, 633 nm).
    
    :math:`g(\lambda) = 0.29 \times 10^{-3} \lambda + 0.62`
    
    | wavelength range: [302 nm, 633 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **g** (*float or np.ndarray*) – anisotropy factor [-]
    '''
    
    return linear(lambda0, 0.29E-3, 0.62)

def g_Bosschaart(lambda0):
    r'''
    | The anisotropy factor of OXYGENATED BLOOD as a function of wavelength.
    | Linear interpolation of experimental data compiled by Bosschaart et al. 2014 [B*14].
    
    wavelength range: [251 nm, 1000 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **g** (*float or np.ndarray*) – anisotropy factor [-]
    '''
    
    return interp1d(np.array(oxy_and_deo_Bosschaart_dataframe)[:,0], 
                    np.array(oxy_and_deo_Bosschaart_dataframe)[:,4],
                    bounds_error = False,
                    fill_value = (np.array(oxy_and_deo_Bosschaart_dataframe)[0,4],
                                  np.array(oxy_and_deo_Bosschaart_dataframe)[-1,4]))(lambda0)
