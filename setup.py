from setuptools import find_packages, setup
import codecs

# from setup_backup import LONG_DESCRIPTION

DISTNAME = "qolmat"
VERSION = "0.0.3"
DESCRIPTION = "Tools to impute"
LONG_DESCRIPTION_CONTENT_TYPE = "text/x-rst"
with codecs.open("README.rst", encoding="utf-8-sig") as f:
    LONG_DESCRIPTION = f.read()
# LONG_DESCRIPTION = "hello"

# """
# Here we should add the correct
# URL = "https://github.com/scikit-learn-contrib/...."
# DOWNLOAD_URL = "https://pypi.org/project/......"
# PROJECT_URLS = {
#     "Bug Tracker": "https://github.com/scikit-learn-contrib/...../issues",
#     "Documentation": "https://......readthedocs.io/en/latest/",
#     "Source Code": "https://github.com/scikit-learn-contrib/......"
# }
# """

LICENSE = "new BSD"
AUTHORS = "Hong-Lan Botterman, Julien Roussel, Thomas Morzadec, Rima Hajou"
AUTHORS_EMAIL = """
hlbotterman@quantmetry.com,
jroussel@quantmetry.com,
tmorzadec@quantmetry.com,
rhajou@quantmetry.com
"""

PYTHON_REQUIRES = ">=3.8"
PACKAGES = find_packages()
INSTALL_REQUIRES = ["scikit-learn", "numpy>=1.21", "packaging"]
EXTRAS_REQUIRE = {
    "tests": ["flake8", "mypy", "pandas", "pytest", "pytest-cov", "typed-ast"],
    "docs": [
        "matplotlib",
        "numpydoc",
        "pandas",
        "sphinx",
        "sphinx-gallery",
        "sphinx_rtd_theme",
        "typing_extensions",
    ],
}

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

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
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=CLASSIFIERS,
    zip_safe=False,
)
