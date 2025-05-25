# setup.py
from setuptools import setup
from Cython.Build import cythonize
import glob, numpy

pyx_files = glob.glob("algorithm/**/*.pyx", recursive=True)

setup(
    name="my_project",
    ext_modules=cythonize(
        pyx_files,
        include_path=[numpy.get_include()],
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
