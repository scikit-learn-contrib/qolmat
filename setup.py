from setuptools import setup, find_packages

DISTNAME = "qolmat"
VERSION_FILE = "qolmat/_version.py"
with open(VERSION_FILE, "rt") as f:
    version_txt = f.read().strip()
    VERSION = version_txt.split('"')[1]
DESCRIPTION = "Tools to compute RPCA"
LONG_DESCRIPTION = "Tools to compute different formulations of RPCA for matrices and times series."
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
LICENSE = "new BSD"
AUTHORS = "Hong-Lan Botterman, Julien Roussel, Thomas Morzadec"
AUTHORS_EMAIL = """
hlbotterman@quantmetry.com,
jroussel@quantmetry.com,
tmorzadec@quantmetry.com
"""
PACKAGES = find_packages()
INSTALL_REQUIRES = [
    "numpy",
    "pandas",
    "matplotlib",
    "plotly",
    "scikit-optimize",
    "scipy",
    "tqdm",
    "pillow",
    "scikit-learn",
]
PYTHON_REQUIRES = ">=3.8"

setup(
    name=DISTNAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHORS,
    author_email=AUTHORS_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
)


# INSTALL_REQUIRES = [
#     "numpy>=1.22.1",
#     "pandas>=1.4.0",
#     "matplotlib>=3.5.1",
#     "plotly>=5.5.0",
#     "scikit-optimize>=0.9.0",
#     "scipy>=1.7.3",
#     "tqdm>=4.62.3",
#     "pillow>=8.4.0",
#     "scikit-learn>=1.0.2"
# ]
