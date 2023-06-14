# Building an extension module from C source
from distutils.core import setup, Extension
import numpy

simulation_mod = Extension('simulations',
                              sources = ['simulation_wilson.cpp'],
                              language="c++",
                              include_dirs=[numpy.get_include(),
                              "C:\\Users\\shahi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\numpy\\core\\include",
                              "C:\\cpp_libs\\include",
                              "C:\\cpp_libs\\include\\bayesopt\\include",
                              "C:\\src\\vcpkg\\installed\\x64-windows\\include",
                              "C:\\src\\vcpkg\\installed\\x86-windows\\include",
                              "C:\\Users\\shahi\\AppData\\Local\\Programs\\Python\\Python311\\include",
                              "C:\\Users\\shahi\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages"
                              ],
                              library_dirs=[
                                  "C:\\src\\vcpkg\\installed\\x64-windows\\lib",
                                  "C:\\src\\vcpkg\\installed\\x86-windows\\lib",
                                  "C:\\msys64\\mingw64\\bin",
                                  "C:\\cpp_libs\\include\\bayesopt\\build_msvc\\lib\\Release"
                              ],
                              libraries=[
                              "gsl",
                              "gslcblas",
                              "bayesopt",
                              "nlopt"
                              ]
                            )

# The main setup command
setup(name = 'DissertationModels',
      version="1.0",
      description="Simulation models and temporal integration for the Wilson Cowan model",
      ext_modules=[simulation_mod],
      # py_modules=['simulation_interface'],
)                            