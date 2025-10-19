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
| victorporto\@ifsc.usp.br
| victor.lima\@ufscar.br
| victorportog.github.io

| Release date:
| October 2024
| Last modification:
| October 2024

| References:

| [HQ73] Hale & Querry 1973.
| Optical Constants of Water in the 200-nm to 200-μm Wavelength Region.
| https://doi.org/10.1364/AO.12.000555

| [S81] Segelstein 1981.
| The complex refractive index of water.
| https://mospace.umsystem.edu/xmlui/handle/10355/11599

| [AF90] Agati & Fusi 1990.
| New trends in photobiology recent advances in bilirubin photophysics.
| https://doi.org/10.1016/1011-1344(90)85138-M

| [B90] Billett 1990.
| Hemoglobin and Hematocrit.
| https://www.ncbi.nlm.nih.gov/books/NBK259/

| [v*05] van Veen, Sterenborg, Pifferi, Torricelli, Chikoidze & Cubeddu 2005.
| Determination of visible near-IR absorption coefficients of mammalian fat using time- and spatially resolved
| diffuse reflectance and transmission spectroscopy.
| https://doi.org/10.1117/1.2085149

| [H02] Hecht 2002.
| Optics. (4th Edition)

| [S*06] Salomatina, Jiang, Novak & Yaroslavsky 2006.
| Optical properties of normal and cancerous human skin in the visible and near-infrared spectral range.
| https://doi.org/10.1117/1.2398928

| [SS06] Sarna & Swartz 2006.
| The Physical Properties of Melanins.
| https://doi.org/10.1002/9780470987100.ch16

| [DJV11] Delgado Atencio, Jacques & Vázquez y Montiel 2011.
| Monte Carlo Modeling of Light Propagation in Neonatal Skin.
| https://doi.org/10.5772/15853

| [J13] Jacques 2013.
| Optical properties of biological tissues: a review.
| https://doi.org/10.1088/0031-9155/58/14/5007

| [B*14] Bosschaart, Edelman, Aalders, van Leeuwen & Faber 2014.
| A literature review and novel theoretical approach on the optical properties of whole blood.
| https://doi.org/10.1007/s10103-013-1446-7

| [G17] Griffiths 2017.
| Introduction to Electrodynamics. (4th Edition)
| https://doi.org/10.1017/9781108333511

| [TL23] Taniguchi & Lindsey 2023.
| Absorption and fluorescence spectra of open-chain tetrapyrrole pigments – bilirubins, biliverdins, phycobilins, and synthetic analogues.
| https://doi.org/10.1016/j.jphotochemrev.2023.100585

| [S*23] Sá, Bacal, Gomes, Silva, Gonçalves & Malta 2023.
| Blood count reference intervals for the Brazilian adult population: National Health Survey.
| https://doi.org/10.1590/1980-549720230004.supl.1
'''

import numpy as np
from scipy.interpolate import interp1d

from skinoptics.utils import *
from skinoptics.dataframes import *


def Cmass_from_Cmolar(Cmolar, molar_mass):
    r'''
    Calculate the mass concentration from the molar concentration and the molar mass.

    :math:`C_{mass} = M \mbox{ } C_{molar}`

    :param Cmolar: molar concentration [M]
    :type Cmolar: float or np.ndarray

    :param molar_mass: molar mass [g mol^-1]
    :type molar_mass: float
    
    :return: - **Cmass** (*float or np.ndarray*) – mass concentration [g L^-1]
    '''

    return Cmolar*molar_mass
    
def Cmolar_from_Cmass(Cmass, molar_mass):
    r'''
    Calculate the molar concentration from the mass concentration and the molar mass.

    :math:`C_{molar} = \frac{C_{mass}}{M}`
 
    :param Cmass: mass concentration [g L^-1]
    :type Cmass: float or np.ndarray

    :param molar_mass: molar mass [g mol^-1]
    :type molar_mass: float

    :return: - **Cmolar** (*float or np.ndarray*) – molar concentration [M]
    '''

    return Cmass/molar_mass

def ext_from_Abs_and_Cmass(Abs, Cmass, pathlength):
    r'''
    | Calculate the extinction coefficient from the absorbance, the mass concentration
    | and the pathlength.

    :math:`\varepsilon_{mass}(\lambda) = \frac{Abs(\lambda)}{L \mbox{ } C_{mass}}`
    
    :param Abs: absorbance [-]
    :type Abs: float or np.ndarray
    
    :param Cmass: mass concentration [g L^-1]
    :type Cmass: float
    
    :param pathlength: pathlength [cm]
    :type pathlength: float
    
    :return: - **ext** (*float or np.ndarray*) – extinction coefficient [cm^-1 mL mg^-1]
    '''
    
    return Abs/Cmass/pathlength

def molarext_from_Abs_and_Cmolar(Abs, Cmolar, pathlength):
    r'''
    | Calculate the molar extinction coefficient from the absorbance, the molar concentration
    | and the pathlength.
    
    :math:`\varepsilon_{molar}(\lambda) = \frac{Abs(\lambda)}{L \mbox{ } C_{molar}}`
    
    :param Abs: absorbance [-]
    :type Abs: float or np.ndarray
    
    :param Cmolar: molar concentration [g L^-1]
    :type Cmolar: float
    
    :param pathlength: pathlength [cm]
    :type pathlength: float

    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return Abs/Cmolar/pathlength

def ext_from_molarext(molarext, molar_mass):
    r'''
    | Calculate the extinction coefficient from the molar extinction coefficient
    | and the molar mass.

    :math:`\varepsilon_{mass}(\lambda) = \frac{\varepsilon_{molar}(\lambda)}{M}`
    
    :param molarext: molar extinction coefficient [cm^-1 M^-1]
    :type molarext: float or np.ndarray
    
    :param molar_mass: molar mass [g mol^-1]
    :type molar_mass: float
    
    :return: - **ext** (*float or np.ndarray*) – extinction coefficient [cm^-1 mL mg^-1]
    '''
        
    return molarext/molar_mass

def molarext_from_ext(ext, molar_mass):
    r'''
    | Calculate the molar extinction coefficient from the extinction coefficient
    | and the molar mass.

    :math:`\varepsilon_{molar}(\lambda) = M \mbox{ } \varepsilon_{mass}(\lambda)`
    
    :param ext: extinction coefficient [cm^-1 mL mg^-1]
    :type ext: float or np.ndarray
    
    :param molar_mass: molar mass [g mol^-1]
    :type molar_mass: float
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return ext*molar_mass
    
def mua_from_ext_and_Cmass(ext, Cmass):
    r'''
    | Calculate the absorption coefficient from the extinction coefficient and the mass concentration.
    | For details please check Jacques 2013 [J13].

    :math:`\mu_a(\lambda) = \mbox{ln}(10) \mbox{ } C_{mass} \mbox{ } \varepsilon_{mass}(\lambda)`
    
    :param ext: extinction coefficient [cm^-1 mL mg^-1]
    :type ext: float or np.ndarray
    
    :param Cmass: mass concentration [g L^-1]
    :type Cmass: float
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
        
    return np.log(10)*ext*Cmass/10

def mua_from_molarext_and_Cmolar(molarext, Cmolar):
    r'''
    | Calculate the absorption coefficient from the molar extinction coefficient
    | and the molar concentration.
    | For details please check Jacques 2013 [J13].

    :math:`\mu_a(\lambda) = \mbox{ln}(10) \mbox{ } C_{molar} \mbox{ } \varepsilon_{molar}(\lambda)`
    
    :param molarext: molar extinction coefficient [cm^-1 M^-1]
    :type molarext: float or np.ndarray
    
    :param Cmolar: molar concentration [M]
    :type Cmolar: float
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
        
    return np.log(10)*molarext*Cmolar/10

def ext_from_mua_and_Cmass(mua, Cmass):
    r'''
    | Calculate the extinction coefficient from the absorption coefficient
    | and the mass concentration.
    | For details please check Jacques 2013 [J13].

    :math:`\varepsilon_{mass}(\lambda) = \frac{1}{\mbox{ln}(10)}\frac{\mu_a(\lambda)}{C_{mass}}`
    
    :param mua: absorption coefficient [mm^-1]
    :type mua: float or np.ndarray
    
    :param Cmass: mass concentration [g L^-1]
    :type Cmass: float
    
    :return: - **ext** (*float or np.ndarray*) – extinction coefficient [cm^-1 mL mg^-1]
    '''
        
    return 10*mua/np.log(10)/Cmass

def molarext_from_mua_Cmolar(mua, Cmolar):
    r'''
    | Calculate the molar extinction coefficient from the absorption coefficient
    | and the molar concentration.
    | For details please check Jacques 2013 [J13].

    :math:`\varepsilon_{molar}(\lambda) = \frac{1}{\mbox{ln}(10)}\frac{\mu_a(\lambda)}{C_{molar}}`
    
    :param mua: absorption coefficient [mm^-1]
    :type mua: float or np.ndarray
    
    :param Cmolar: molar concentration [M]
    :type Cmolar: float
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return 10*mua/np.log(10)/Cmolar

def mua_from_k(k, lambda0):
    r'''
    | Calculate the absorption coefficient from the imaginary part of the complex refractive index
    | and the wavelength.
    | For details please check Hecht 2002 [H02], Jacques 2013 [J13] and Griffiths 2017 [G17].
    
    :math:`\mu_a(\lambda) = 4\pi \frac{k(\lambda)}{\lambda}`
    
    :param k: imaginary part of the complex refractive index [-]
    :type k: float or np.ndarray
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return 4*np.pi*k/lambda0/1E-6

def k_from_mua(mua, lambda0):
    r'''
    | Calculate the imaginary part of the complex refractive index from the absorption coefficient
    | and the wavelength.
    | For details please check Hecht 2002 [H02], Jacques 2013 [J13] and Griffiths 2017 [G17].
    
    :math:`k(\lambda) = \frac{\mu_a(\lambda)}{4\pi}\lambda`
    
    :param mua: absorption coefficient [mm^-1]
    :type mua: float or np.ndarray
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **k** (*float or np.ndarray*) – imaginary part of the complex refractive index [-]
    '''
    return mua*lambda0*1E-6/4/np.pi

def k_wat_Hale(lambda0):
    r'''
    | The imaginary part of the complex refractive index of WATER as a function of wavelength.
    | Linear interpolation of data from Hale & Querry 1973 [HQ73] (see their Table I).
    
    | wavelength range: [200 nm, 200 μm]
    | temperature: 25 ºC
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **k** (*float or np.ndarray*) – imaginary part of the complex refractive index [-]
    '''
            
    return interp1d(np.array(n_and_k_wat_Hale_dataframe)[:,0],
                    np.array(n_and_k_wat_Hale_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(n_and_k_wat_Hale_dataframe)[0,1],
                                  np.array(n_and_k_wat_Hale_dataframe)[-1,1]))(lambda0)

def k_wat_Segelstein(lambda0):
    r'''
    | The imaginary part of the complex refractive index of WATER as a function of wavelength.
    | Linear interpolation of data from D. J. Segelstein's M.S. Thesis 1981 [S81] collected
    | by S. Prahl and publicly available at <https://omlc.org/spectra/water/abs/index.html>.
    
    | wavelength range: [10 nm, 10 m]
        
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **k** (*float or np.ndarray*) – imaginary part of the complex refractive index [-]
    '''
    
    return interp1d(np.array(n_and_k_wat_Segelstein_dataframe)[:,0]*1E3,
                    np.array(n_and_k_wat_Segelstein_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(n_and_k_wat_Segelstein_dataframe)[0,2],
                                  np.array(n_and_k_wat_Segelstein_dataframe)[-1,2]))(lambda0)

def ext_eum_Sarna(lambda0):
    r'''
    | The extinction coefficient of EUMELANIN in phosphate buffer as a function of wavelength.
    | Linear interpolation of data from Sarna & Swartz 2006 [SS06] (see their Fig. 16.3-a)
    | graphically deduced by Jacques and publicly available at
    | <https://omlc.org/spectra/melanin/extcoeff.html>.
    
    | wavelength range: [210 nm, 820 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **ext** (*float or np.ndarray*) – extinction coefficient [cm^-1 mL mg^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_eum_Sarna_dataframe)[:,0],
                    np.array(ext_and_molarext_eum_Sarna_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_eum_Sarna_dataframe)[0,1],
                                  np.array(ext_and_molarext_eum_Sarna_dataframe)[-1,1]))(lambda0)

def molarext_eum_Sarna(lambda0):
    r'''
    | The molar extinction coefficient of EUMELANIN in phosphate buffer as a function of wavelength.
    | Linear interpolation of data from Sarna & Swartz 2006 [SS06] (see their Fig. 16.3-a)
    | graphically deduced by Jacques and publicly available at
    | <https://omlc.org/spectra/melanin/extcoeff.html>.
    
    | wavelength range: [210 nm, 820 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_eum_Sarna_dataframe)[:,0],
                    np.array(ext_and_molarext_eum_Sarna_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_eum_Sarna_dataframe)[0,2],
                                  np.array(ext_and_molarext_eum_Sarna_dataframe)[-1,2]))(lambda0)

def ext_phe_Sarna(lambda0):
    r'''
    | The extinction coefficient of PHEOMELANIN in phosphate buffer as a function of wavelength.
    | Linear interpolation of data from Sarna & Swartz 2006 [SS06] (see their Fig. 16.3-a)
    | graphically deduced by Jacques and publicly available at
    | <https://omlc.org/spectra/melanin/extcoeff.html>.
    
    | wavelength range: [210 nm, 820 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **ext** (*float or np.ndarray*) – extinction coefficient [cm^-1 mL mg^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_phe_Sarna_dataframe)[:,0],
                    np.array(ext_and_molarext_phe_Sarna_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_phe_Sarna_dataframe)[0,1],
                                  np.array(ext_and_molarext_phe_Sarna_dataframe)[-1,1]))(lambda0)

def molarext_phe_Sarna(lambda0):
    r'''
    | The molar extinction coefficient of PHEOMELANIN in phosphate buffer as a function of wavelength.
    | Linear interpolation of data from Sarna & Swartz 2006 [SS06] (see their Fig. 16.3-a)
    | graphically deduced by Jacques and publicly available at
    | <https://omlc.org/spectra/melanin/extcoeff.html>.
    
    | wavelength range: [210 nm, 820 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_phe_Sarna_dataframe)[:,0],
                    np.array(ext_and_molarext_phe_Sarna_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_phe_Sarna_dataframe)[0,2],
                                  np.array(ext_and_molarext_phe_Sarna_dataframe)[-1,2]))(lambda0)

def molarext_oxy_Prahl(lambda0):
    r'''
    | The molar extinction coefficient of OXYHEMOGLOBIN in water as a function of wavelength.
    | Linear interpolation of data from various sources compiled by S. Prahl and publicly
    | available at <https://omlc.org/spectra/hemoglobin/>.
    
    | wavelength range: [250 nm, 1000 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[:,0],
                    np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[0,1],
                                  np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[-1,1]))(lambda0)

def molarext_deo_Prahl(lambda0):
    r'''
    | The molar extinction coefficient for DEOXYHEMOGLOBIN in water as a function of wavelength.
    | Linear interpolation of data from various sources compiled by S. Prahl and publicly
    | available at <https://omlc.org/spectra/hemoglobin/>.
    
    | wavelength range: [250 nm, 1000 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return interp1d(np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[:,0],
                    np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[0,2],
                                  np.array(ext_and_molarext_oxy_and_deo_Prahl_dataframe)[-1,2]))(lambda0)

def molarext_bil_Li(lambda0):
    r'''
    | The molar extinction coefficient of BILIRUBIN in chloroform as a function of wavelength.
    | Linear interpolation of experimental data obtained with a Cary 3 by J. Li on 1997,
    | scaled to match 55,000 cm^-1 M^-1 at 450.8 nm [AF90] and publicly available by S. Prahl
    | at <https://omlc.org/spectra/PhotochemCAD/html/119.html>.
    | The data is also available at PhotochemCAD [TL23]
    | <https://www.photochemcad.com/databases/common-compounds/oligopyrroles/bilirubin>.
    
    | wavelength range: [239.75 nm, 700 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **molarext** (*float or np.ndarray*) – molar extinction coefficient [cm^-1 M^-1]
    '''
    
    return interp1d(np.array(molarext_bil_Li_dataframe)[:,0], 
                    np.array(molarext_bil_Li_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(molarext_bil_Li_dataframe)[0,1],
                                  np.array(molarext_bil_Li_dataframe)[-1,1]))(lambda0)
            
def mua_baseline(lambda0):
    r'''
    | The baseline absorption coefficient as a function of wavelength.
    | Equation proposed by S. Jacques based on data for bloodless rat skin.
    | For details please check <https://omlc.org/news/jan98/skinoptics.html>.
    
    :math:`\mu_a^{bas}(\lambda) = 0.0244 + 8.53\mbox{ exp}(-(\lambda-154)/66.2)`

    | wavelength range: [350 nm, 1100 nm]

    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return (0.244 + 85.3*np.exp(-(lambda0 - 154.)/66.2))/10

def mua_baseline2(lambda0):
    r'''
    | The baseline absorption coefficient as a function of wavelength.
    | Equation based on data for neonatal skin.
    | For details please check <https://omlc.org/news/jan98/skinoptics.html>.
    
    :math:`\mu_a^{bas}(\lambda) = 7.84 \times 10^7 \times \lambda^{-3.255}`

    | wavelength range: [450 nm, 750 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return 7.84E7*lambda0**(-3.255)        
        
def mua_wat_Hale(lambda0):
    r'''
    | The absorption coefficient of WATER as a function of wavelength.
    | Linear interpolation of data from Hale & Querry 1973 [HQ73] collected and processed 
    | by S. Prahl and publicly available at <https://omlc.org/spectra/water/abs/index.html>.
    
    wavelength range: [200 nm, 200 μm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(mua_wat_Hale_dataframe)[:,0],
                    np.array(mua_wat_Hale_dataframe)[:,1]/10,
                    bounds_error = False,
                    fill_value = (np.array(mua_wat_Hale_dataframe)[0,1]/10,
                                  np.array(mua_wat_Hale_dataframe)[-1,1]/10))(lambda0)

def mua_wat_Segelstein(lambda0):
    r'''
    | The absorption coefficient of WATER as a function of wavelength.
    | Linear interpolation of data from D. J. Segelstein's M.S. Thesis 1981 [S81],
    | collected by S. Prahl and publicly available at <https://omlc.org/spectra/water/abs/index.html>.
    
    | wavelength range: [10 nm, 10 m].
        
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(mua_wat_Segelstein_dataframe)[:,0],
                    np.array(mua_wat_Segelstein_dataframe)[:,1]/10,
                    bounds_error = False,
                    fill_value = (np.array(mua_wat_Segelstein_dataframe)[0,1]/10,
                                  np.array(mua_wat_Segelstein_dataframe)[-1,1]/10))(lambda0)
    
def mua_mel_Jacques(lambda0):
    r'''
    | The absoption coefficient of a MELANOSOME as a function of wavelength.
    | Equation proposed by S. Jacques based on data from various sources.
    | For details please check <https://omlc.org/news/jan98/skinoptics.html>.

    :math:`\mu_a^{mel} (\lambda) = 6.6 \times 10^{10} \times \lambda^{-3.33}`
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return 6.6E10*lambda0**(-3.33) 

def mua_oxy_Prahl(lambda0, Cmass_oxy = 150, molar_mass_oxy = 64500):
    r'''
    | The absorption coefficient of OXYHEMOGLOBIN in water as a function of wavelength.
    | Calculated from :meth:`skinoptics.absorption_coefficient.molarext_oxy_Prahl` for a specific
    | oxyhemoglobin mass concentration.
    
    | wavelength range: [250 nm, 1000 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param Cmass_oxy: oxyhemoglobin mass concentration [g/L] (default to 150. [S*23])
    :type Cmass_oxy: float
    
    :param molar_mass_oxy: molar mass of oxyhemoglobin [g/mol] (default  to 64500. [B90])
    :type molar_mass_oxy: float
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return mua_from_molarext_and_Cmolar(molarext_oxy_Prahl(lambda0),
                                        Cmolar = Cmolar_from_Cmass(Cmass = Cmass_oxy, molar_mass = molar_mass_oxy))

def mua_deo_Prahl(lambda0, Cmass_deo = 150, molar_mass_deo = 64500):
    r'''
    | The absorption coefficient of DEOXYHEMOGLOBIN in water as a function of wavelength.
    | Calculated from :meth:`skinoptics.absorption_coefficient.molarext_deo_Prahl` for a specific
    | deoxyhemoglobin mass concentration.
    
    | wavelength range: [250 nm, 1000 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray.
    
    :param Cmass_oxy: deoxyhemoglobin mass concentration [g/L] (default to 150. [S*23])
    :type Cmass_oxy: float
    
    :param molar_mass_deo: molar mass of deoxyhemoglobin [g/mol] (default to 64500. [B90])
    :type molar_mass_deo: float
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return mua_from_molarext_and_Cmolar(molarext_deo_Prahl(lambda0),
                                        Cmolar = Cmolar_from_Cmass(Cmass = Cmass_deo, molar_mass = molar_mass_deo))

def mua_oxy_Bosschaart(lambda0):
    r'''
    | The absorption coefficient of OXYGENIZED BLOOD (saturation > 98%) as a function of wavelength.
    | Linear interpolation of data from Bosschaart et. al. 2014 [B*14].
    
    | wavelength range: [251 nm, 1995 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(oxy_and_deo_Bosschaart_dataframe)[:,0],
                    np.array(oxy_and_deo_Bosschaart_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(oxy_and_deo_Bosschaart_dataframe)[0,1],
                                  np.array(oxy_and_deo_Bosschaart_dataframe)[-1,1]))(lambda0)

def mua_deo_Bosschaart(lambda0):
    r'''
    | The absorption coefficient of DEOXIGENIZED BLOOD (saturation = 0%) as a function of wavelength.
    | Linear interpolation of data from Bosschaart et. al. 2014 [B*14].
    
    | wavelength range: [251 nm, 1995 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(oxy_and_deo_Bosschaart_dataframe)[:,0],
                    np.array(oxy_and_deo_Bosschaart_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(oxy_and_deo_Bosschaart_dataframe)[0,2],
                                  np.array(oxy_and_deo_Bosschaart_dataframe)[-1,2]))(lambda0)

def mua_bil_Li(lambda0, Cmass_bil, molar_mass_bil = 585):
    r'''
    | The absorption coefficient of BILIRUBIN in chloroform as a function of wavelength.
    | Calculated from :meth:`skinoptics.absorption_coefficient.molarext_bil_Li` for a specific
    | bilirubin mass concentration.

    | wavelength range: [239.75 nm, 700 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :param Cmass_bil: bilirubin mass concentration [g/L]
    :type Cmass_bil: float
    
    :param molarmass_bil: molar mass of bilirubin [g/mol] (default to 585. [DJV11])
    :type molarmass_bil: float
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return mua_from_molarext_and_Cmolar(molarext_bil_Li(lambda0),
                                        Cmolar = Cmolar_from_Cmass(Cmass = Cmass_bil, molar_mass = molar_mass_bil))

def mua_fat_vanVeen(lambda0):
    r'''
    | The absorption coefficient of (pig lard) FAT as a function of wavelength.
    | Linear interpolation of data from van Veen et al. 2005 [v*05] collected and processed
    | by S. Prahl and publicly available at <https://omlc.org/spectra/fat/>.
    
    | wavelength range: [429 nm, 1098 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray

    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(mua_fat_vanVeen_dataframe)[:,0],
                    np.array(mua_fat_vanVeen_dataframe)[:,1]/1E3,
                    bounds_error = False,
                    fill_value = (np.array(mua_fat_vanVeen_dataframe)[0,1]/1E3,
                                  np.array(mua_fat_vanVeen_dataframe)[-1,1]/1E3))(lambda0)

def mua_EP_Salomatina(lambda0):
    r'''
    | The absoption coefficient of human EPIDERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(EP_Salomatina_dataframe)[:,0], 
                    np.array(EP_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(EP_Salomatina_dataframe)[0,1],
                                  np.array(EP_Salomatina_dataframe)[-1,1]))(lambda0)

def mua_DE_Salomatina(lambda0):
    r'''
    | The absoption coefficient of human DERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(DE_Salomatina_dataframe)[:,0],
                    np.array(DE_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(DE_Salomatina_dataframe)[0,1],
                                  np.array(DE_Salomatina_dataframe)[-1,1]))(lambda0)

def mua_HY_Salomatina(lambda0):
    r'''
    | The absoption coefficient of human HYPODERMIS as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [374 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(HY_Salomatina_dataframe)[:,0],
                    np.array(HY_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(HY_Salomatina_dataframe)[0,1],
                                  np.array(HY_Salomatina_dataframe)[-1,1]))(lambda0)

def mua_iBCC_Salomatina(lambda0):
    r'''
    | The absoption coefficient of INFILTRATIVE BASAL CELL CARCINOMA as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    return interp1d(np.array(iBCC_Salomatina_dataframe)[:,0],
                    np.array(iBCC_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(iBCC_Salomatina_dataframe)[0,1],
                                  np.array(iBCC_Salomatina_dataframe)[-1,1]))(lambda0)

def mua_nBCC_Salomatina(lambda0):
    r'''
    | The absoption coefficient of NODULAR BASAL CELL CARCINOMA as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(nBCC_Salomatina_dataframe)[:,0],
                    np.array(nBCC_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(nBCC_Salomatina_dataframe)[0,1],
                                  np.array(nBCC_Salomatina_dataframe)[-1,1]))(lambda0)

def mua_SCC_Salomatina(lambda0):
    r'''
    | The absoption coefficient of SQUAMOUS CELL CARCINOMA as a function of wavelength.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.
    
    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **mua** (*float or np.ndarray*) – absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(SCC_Salomatina_dataframe)[:,0],
                    np.array(SCC_Salomatina_dataframe)[:,1],
                    bounds_error = False,
                    fill_value = (np.array(SCC_Salomatina_dataframe)[0,1],
                                  np.array(SCC_Salomatina_dataframe)[-1,1]))(lambda0)

def std_mua_EP_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.absorption_coefficient.mua_EP_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.

    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_mua** (*float or np.ndarray*) – standard deviation of the absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(EP_Salomatina_dataframe)[:,0], 
                    np.array(EP_Salomatina_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(EP_Salomatina_dataframe)[0,2],
                                  np.array(EP_Salomatina_dataframe)[-1,2]))(lambda0)

def std_mua_DE_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.absorption_coefficient.mua_DE_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.

    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_mua** (*float or np.ndarray*) – standard deviation of the absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(DE_Salomatina_dataframe)[:,0], 
                    np.array(DE_Salomatina_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(DE_Salomatina_dataframe)[0,2],
                                  np.array(DE_Salomatina_dataframe)[-1,2]))(lambda0)

def std_mua_HY_Salomatina(lambda0):
    r'''
    | The standard deviation respective to :meth:`skinoptics.absorption_coefficient.mua_HY_Salomatina`.
    | Linear interpolation of experimental data from Salomatina et al. 2006 [S*06],
    | publicly available at <https://sites.uml.edu/abl/optical-properties-2/>.

    | wavelength range: [370 nm, 1600 nm]
    
    :param lambda0: wavelength [nm]
    :type lambda0: float or np.ndarray
    
    :return: - **std_mua** (*float or np.ndarray*) – standard deviation of the absorption coefficient [mm^-1]
    '''
    
    return interp1d(np.array(HY_Salomatina_dataframe)[:,0], 
                    np.array(HY_Salomatina_dataframe)[:,2],
                    bounds_error = False,
                    fill_value = (np.array(HY_Salomatina_dataframe)[0,2],
                                  np.array(HY_Salomatina_dataframe)[-1,2]))(lambda0)
