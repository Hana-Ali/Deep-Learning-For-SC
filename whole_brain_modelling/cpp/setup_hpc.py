# Building an extension module from C source
from distutils.core import setup, Extension
import numpy

simulation_mod = Extension('simulations',
                              sources = ['simulation_models.cpp'],
                              language="c++",
                              include_dirs=[numpy.get_include(),
                                "/sw-eb/software/Boost/1.81.0-GCC-12.2.0/include",
                                "/sw-eb/software/GSL/2.7-GCC-12.2.0/include "
                              ],
                              library_dirs=[
                                  "/sw-eb/software/Boost/1.81.0-GCC-12.2.0/lib",
                                  "/sw-eb/software/GSL/2.7-GCC-12.2.0/lib"
                              ],
                              libraries=[
                              "gsl",
                              "gslcblas",
                              ]
                            )

# The main setup command
setup(name = 'DissertationModels',
      version="1.0",
      description="Simulation models and temporal integration for the Wilson Cowan model",
      ext_modules=[simulation_mod],
      # py_modules=['simulation_interface'],
)                            