from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy", 
    "pandas", 
    "matplotlib",
    "plotly",
    "scikit-optimize",
    "scipy",
    "tqdm",
    "pillow",
    "scikit-learn"
]
PYTHON_REQUIRES = ">=3.7"
VERSION_FILE = "robust_pca/_version.py"
with open(VERSION_FILE, "rt") as f :
    version_txt = f.read().strip() 
    VERSION = version_txt.split('"')[1]

setup(
    name="robust_pca",
    version=VERSION,
    license="new BSD",
    author="Hong-Lan Botterman",
    author_email="hlbotterman@quantmetry.com",
    description="Tools to compute RPCA",
    long_description="Tools to compute different formulations of RPCA for matrices and times series.",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
)