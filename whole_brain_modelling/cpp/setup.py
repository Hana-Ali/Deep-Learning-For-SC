# Building an extension module from C source
from distutils.core import setup, Extension
import numpy

simulation_mod = Extension('simulations',
                              sources = ['simulation_models.cpp'],
                              language="c++",
                              include_dirs=[
                                numpy.get_include(),
                                "C:\\src\\vcpkg\\installed\\x64-windows\\include"
                              ],
                              library_dirs=[
                                  "C:\\src\\vcpkg\\installed\\x64-windows\\lib"
                              ],
                              libraries=[
                              "gsl",
                              "gslcblas"
                              ]
                            )

# The main setup command
setup(name = 'DissertationModels',
      version="1.0",
      description="Simulation models and temporal integration for the Wilson Cowan model",
      ext_modules=[simulation_mod],
      # py_modules=['simulation_interface'],
)                            