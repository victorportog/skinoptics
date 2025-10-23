SkinOptics documentation
========================

| **SkinOptics** is an open source Python package with tools for building human skin computational
| models for Monte Carlo simulations of light transport, as well as tools for analyzing simulation
| outputs. It can also be used for teaching and exploring about Optical Properties and Colorimetry.

**SkinOptics** is under continuous development.

New features may be available in the future.

Please remember that **SkinOptics** is available without any warranty.

API reference (Toolbox)
-----------------------

.. toctree::
   :maxdepth: 1

   modules

Installation
------------

| **SkinOptics** can be easily installed using ``pip`` (the package installer for Python).
| First of all, please remember to update ``pip`` by running the following line in the command prompt:

.. code-block:: console

      $ pip install --upgrade pip

| Then, to install **SkinOptics**, please use the following command:

.. code-block:: console

      $ pip install skinoptics

| If you have already installed the **SkinOptics** Python package and want to update it to the latest released version, please execute the line:

.. code-block:: console

      $ pip install --upgrade skinoptics

Prerequisite Packages
---------------------

| **SkinOptics** is structured based on other Python packages.
| In order to be able to properly run it, please also install all of the following:

- `NumPy <https://numpy.org/>`_ (>=2.3.4, <3.0.0)
- `SciPy <https://scipy.org/>`_ (>=1.13.0, <2.0.0)
- `pandas <https://pandas.pydata.org/>`_ (>=2.2.2, <3.0.0)

Modules
-------

**SkinOptics** is currently composed of seven modules:

| 1- **utils.py**
| Module with mathematical and auxiliary functions.

| 2- **dataframes.py**
| Module with pandas DataFrames of the txt and csv files stored at the datasets folder.

| 3- **absorption_coefficient.py**
| Module with functions for modeling the absorption coefficient and its related quantities.

| 4- **scattering_cofficient.py**
| Module with functions for modeling the scattering coefficient and its related quantities.

| 5- **anisotropy_factor.py**
| Module with functions for modeling the anisotropy factor and its related quantities.

| 6- **refractive_index.py**
| Module with functions for modeling the refractive index and its related quantities.

| 7- **colors.py**
| Module with functions for Colorimetry.

Jupyter Notebooks
-----------------

- `Tutorial - optical properties <https://skinoptics.readthedocs.io/en/latest/_static/tutorial_optical_properties.html>`_
- `Tutorial - colors <https://skinoptics.readthedocs.io/en/latest/_static/tutorial_colors.html>`_
- `Crosscheck - skinoptics.colors and colour-science <https://skinoptics.readthedocs.io/en/latest/_static/crosscheck_colors.html>`_
- `Reproducing results - Delgado Atencio, Jacques & Vázquez y Montiel 2011 <https://skinoptics.readthedocs.io/en/latest/_static/reproducing_2011DelgadoAtencio.html>`_
- `Validation - skinoptics.colors.Delta_E_00 <https://skinoptics.readthedocs.io/en/latest/_static/validation_Delta_E_00.html>`_

Source Code
-----------

**SkinOptics** scripts are publicly available on `GitHub <https://github.com/victorportog/skinoptics>`_.

Release History
---------------
A list of all **SkinOptics** versions, ordered by release date, is available on `PyPI <https://pypi.org/project/skinoptics/#history>`_.

License
-------

**SkinOptics** is released under GNU General Public License v3.0:

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

Citing SkinOptics
-----------------

If **SkinOptics** is being useful for a presentation, publication or other purpose, please consider citing it.

Suggestion:

   | LIMA, V. P. G. SkinOptics documentation. 2024.
   | Available at <https://skinoptics.readthedocs.io/>.

BibTex entry:

   | @misc{Lima2024,
   |  author = {V. P. G. Lima},
   |  title = {{SkinOptics documentation}},
   |  year  = {2024},
   |  url   = {https://skinoptics.readthedocs.io/}
   | }

Credits
-------

| This documentation is written in reStructuredText and was built with `Sphinx <https://www.sphinx-doc.org/en/master/>`_.
| It is displayed using the `Furo Theme <https://github.com/pradyunsg/furo>`_ and is hosted in the plataform `Read the Docs <https://about.readthedocs.com/>`_. 
| The favicon image is from `OpenMoji <https://openmoji.org/>`_.

References
----------

| References that were used to implement **SkinOptics** tools are stated in the `API reference <modules.html>`_.

Author Information
------------------

| Victor Porto Gontijo de Lima
| Brazilian |:green_square:| |:yellow_square:| |:blue_square:| |:white_large_square:|
| Research, Development & Innovation

| Bachelor in Physics
| Institute of Physics (IF)
| University of Brasília (UnB)
| contact e-mail: discontinued

| MSc in Theoretical and Experimental Physics
| São Carlos Institute of Physics (IFSC)
| University of São Paulo (USP)
| contact e-mail: discontinued

| Physicist (Technician)
| Núcleo de Formação de Professores (NFP)
| Federal University of São Carlos (UFSCar) 
| contact e-mail: victor.lima\@ufscar.br

If you find any mistakes in the scripts or have any suggestions, feel free to send me a message! :)

Funding
-------
| The development of **SkinOptics** was only possible due to multiple sources of finance...

| 1- Grants from Brazilian government funding agencies:

- process number 88887.631088/2021-00 (PROEX - CAPES) [MSc scholarship, 2021 - 2023]
- process number 465360/2014-9 (`INCT <https://www.ifsc.usp.br/cepof/>`_ - CNPq) [Biophotonics Group]
- process number 2014/50857-8 (`INCT <https://www.ifsc.usp.br/cepof/>`_ - FAPESP) [Biophotonics Group]
- process number 2013/07276-1 (CePID `CePOF <https://www.ifsc.usp.br/cepof/>`_ - FAPESP) [Biophotonics Group]

| 2- My work with management, teaching, research and university extension as an employee of a higher education federal institution in Brazil:

- currently at the Federal University of São Carlos (UFSCar)
- previously at the Federal Institute of Education, Science and Technology of Brasília (IFB)

Acknowledgments
---------------

I would like to thank...

- my MSc supervisor Lilian Tan Moriyama, who is guiding me throughout the amazing and intriguing field of Biophotonics and Biomedical Optics.

- some colleagues Thereza Fortunato, Otávio Palamoni, Sofia Santos, Semira Silva, Johan Tovar, Maria Júlia Marques and Alessandra Lima, as well as professors Cristina Kurachi and Vanderlei Bagnato, for all aid and discussions.

- my BSc supervisor Mariana Penna Lima Vitenti, who guided my first steps in the field of Scientific Computing.

- my friend Arthur Willian, who always helps me to deal with computer-related issues (and many others).

- all my friends, who I will not be able to list in full here without the risk of missing names and who bring joy even in the hard times.

- my boyfriend Brinatti, with whom I share home and future.

- my grandmas Maria and Eva, my grandpa Severo, my mom Rosana, my dad Silvério and my sister Amanda, who I love so much and who have always provided me with material conditions for living and studying.
