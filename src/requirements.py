from setuptools import setup, find_packages

setup(
    name="PlasMol",
    version="1.1.0",
    author="Brinton Eldridge",
    author_email="bldrdge1@memphis.edu",
    description="A tool for simulating plasmon-molecule interactions using FDTD and RT-TDDFT.",
    long_description=open("README.md").read() if open("README.md", errors="ignore") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kombatEldridge/PlasMol",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pillow",
        "pyscf",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)