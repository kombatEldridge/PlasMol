from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="PlasMol",
    version="1.1.0",
    author="Brinton Eldridge",
    author_email="bldrdge1@memphis.edu",
    description="A tool for simulating plasmon-molecule interactions using FDTD and RT-TDDFT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kombatEldridge/PlasMol",
    packages=find_packages(),
    package_data={
        'PlasMol': ['templates/*.in'],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "pillow>=9.0",
        "pyscf>=2.0",
        # 'meep' is not added here; document in README that it requires conda
    ],
    extras_require={
        "classical": ["meep"],  # Users install with pip install PlasMol[classical]
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=5.0"],  # For testing/formatting
    },
    entry_points={
        "console_scripts": [
            "plasmol = plasmol.main:main",  # Makes 'plasmol' command available after install
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",  # Update to 5 - Production/Stable when ready
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    keywords="plasmon molecule fdtd tddft simulation",
    project_urls={
        "Bug Tracker": "https://github.com/kombatEldridge/PlasMol/issues",
        "Documentation": "https://kombateldridge.github.io/PlasMol",
        "Source Code": "https://github.com/kombatEldridge/PlasMol",
    },
)