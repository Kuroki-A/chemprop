"""Setuptools configuration for the maintained Chemprop v1 fork.

``environment.yml`` is the reproducible CUDA 12.4 environment used by this
repository. The ranges below are intentionally less specific: Python package
metadata must not select a CUDA build of PyTorch, and compatible patch releases
should remain installable.
"""

from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class ChempropBuildPy(build_py):
    """Excludes local comparison-only modules from distributions."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package == "chemprop.features":
            modules = [module for module in modules if module[1] != "_features_generators"]
        return modules


VERSION = "1.7.1+kuroki.3"

# Keep Web assets explicit rather than relying on ``include_package_data`` to
# interpret data-only directories as namespace packages. Newer Setuptools
# warns that the latter is ambiguous and may change behavior in the future.
WEB_APP_ROOT = Path(__file__).parent / "chemprop" / "web" / "app"
WEB_PACKAGE_DATA = [
    path.relative_to(WEB_APP_ROOT).as_posix()
    for directory in (WEB_APP_ROOT / "templates", WEB_APP_ROOT / "static")
    for path in sorted(directory.rglob("*"))
    if path.is_file()
]
WEB_PACKAGE_DATA.append("schema.sql")

CORE_REQUIREMENTS = [
    "Flask>=3.1.3,<3.2",
    "Werkzeug>=3.1.8,<3.2",
    "hyperopt>=0.2.7,<0.4",
    "lightgbm>=4.6,<5",
    "matplotlib>=3.8,<3.11",
    "numpy>=1.26.4,<2.3",
    "packaging>=24.2,<27",
    "pandas>=2.2.3,<2.4",
    "pandas-flavor>=0.6,<0.9",
    "rdkit>=2024.9.6,<2027",
    "scikit-learn>=1.6.1,<1.8",
    "scipy>=1.15.2,<1.16",
    "tensorboardX>=2.6.2,<2.7",
    # CUDA variants cannot be expressed in package metadata. For GPU use,
    # install the desired official PyTorch wheel before installing Chemprop.
    "torch>=2.6,<2.7",
    "tqdm>=4.67,<5",
    "typed-argument-parser>=1.10,<2",
    "typing-extensions>=4.12,<5",
    "descriptastorus>=2.8,<2.9",
]

FEATURE_REQUIREMENTS = [
    "datamol>=0.12.5,<0.13",
    "map4>=1.1.3,<1.2",
    "mhfp>=1.9.6,<2",
    "molfeat>=0.11,<0.12",
    "mordredcommunity>=2.0.7,<2.1",
    "padelpy>=0.1.17,<0.2",
    "pmapper>=1.1.3,<1.2",
]

# Molfeat 0.11's pretrained backends have stricter and partly historical
# dependency requirements. They remain opt-in so the main environment can use
# the newest stable numerical stack without pulling several large frameworks.
PRETRAINED_FEATURE_REQUIREMENTS = [
    "molfeat[dgl,graphormer,fcd,pyg]>=0.11,<0.12",
    "dgl>=1.1.1,<=2.0.0",
    "dgllife>=0.3.2,<0.4",
    "graphormer-pretrained>=0.2.3,<0.3",
    "tokenizers>=0.13,<0.13.2",
    "transformers>=4.24,<4.25",
    "sentencepiece>=0.2,<0.3",
    "selfies>=2.2,<2.3",
]

TEST_REQUIREMENTS = [
    "parameterized>=0.9,<1",
    "pytest>=8.4,<10",
]

DOCS_REQUIREMENTS = [
    "sphinx>=8.1,<8.2",
    "sphinx-rtd-theme>=3.1,<3.2",
]


with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()


setup(
    name="chemprop",
    version=VERSION,
    author="The Chemprop Development Team (see LICENSE.txt)",
    author_email="chemprop@mit.edu",
    description="Molecular Property Prediction with Message Passing Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kuroki-A/chemprop",
    download_url=f"https://github.com/Kuroki-A/chemprop/archive/refs/tags/v{VERSION}.tar.gz",
    project_urls={
        "Documentation": "https://github.com/Kuroki-A/chemprop/tree/master/docs",
        "Source": "https://github.com/Kuroki-A/chemprop",
    },
    license="MIT",
    packages=find_packages(),
    cmdclass={"build_py": ChempropBuildPy},
    include_package_data=False,
    package_data={
        "chemprop": ["py.typed"],
        "chemprop.web.app": WEB_PACKAGE_DATA,
    },
    entry_points={
        "console_scripts": [
            "chemprop_train=chemprop.train:chemprop_train",
            "chemprop_predict=chemprop.train:chemprop_predict",
            "chemprop_fingerprint=chemprop.train:chemprop_fingerprint",
            "chemprop_hyperopt=chemprop.hyperparameter_optimization:chemprop_hyperopt",
            "chemprop_interpret=chemprop.interpret:chemprop_interpret",
            "chemprop_web=chemprop.web.run:chemprop_web",
            "sklearn_train=chemprop.sklearn_train:sklearn_train",
            "sklearn_predict=chemprop.sklearn_predict:sklearn_predict",
        ]
    },
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "test": TEST_REQUIREMENTS,
        "features": FEATURE_REQUIREMENTS,
        "features-pretrained": FEATURE_REQUIREMENTS + PRETRAINED_FEATURE_REQUIREMENTS,
        # Backward-compatible alias for the former all-in-one extra.
        "features-all": FEATURE_REQUIREMENTS + PRETRAINED_FEATURE_REQUIREMENTS,
        "web": ["gunicorn>=25,<27"],
        "scripts": ["h5py>=3.12,<4"],
        "notebooks": ["notebook>=7.5,<8"],
        "docs": DOCS_REQUIREMENTS,
    },
    python_requires=">=3.10,<3.11",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
    ],
    keywords=[
        "chemistry",
        "machine learning",
        "property prediction",
        "message passing neural network",
        "graph neural network",
    ],
)
