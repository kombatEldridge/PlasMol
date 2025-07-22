# About PlasMol

PlasMol is developed for simulating plasmon-molecule interactions, blending classical FDTD with quantum RT-TDDFT.

## Releases

- [**v1.0.0**](https://github.com/kombatEldridge/PlasMol): Initial release containing three main capabilities based on contents of the input file:
    1. **Input contains Nanoparticle (NP) Only**: The team behind [Meep](https://meep.readthedocs.io/en/master/) have built a fantastic codebase with powerful FDTD-based outcomes. When given only parameters surrounding the simulation of a NP, PlasMol acts as a wrapper to Meep, allowing for the simulation of one NP object at a time. As it is the basis of the FDTD implementation for PlasMol, we recommend users visit and use Meep directly, but this is an option in PlasMol for those who want to compare other PlasMol (NP + Molecule) results to an isolated NP simulation. By default, this option only produces real time images of the NP interacting with an electric field. By adding other input flags (described in the docs), PlasMol will track and produce a cross-section extinction spectrum of the NP. Commented sections in the codebase direct users to modify/add functions to track other desired outcomes.
    2. **Input contains Molecule Only**: For inputs with only molecule-based parameters, PlasMol will run a Real-Time Time Dependent Density Functional Theory (RT-TDDFT) simulation. Though more details can be found in the docs, briefly, without any NP present, PlasMol will track the electric field felt by the molecule and the induced dipole moment of the molecule. With some additional flags, PlasMol can produce an absorption spectrum of the molecule.
    3. **Input contains NP and Molecule**: This is the main purpose of PlasMol. A Meep simulation will begin with a molecule inside, whose initial electronic structure is built by PySCF. Every time step, the electric field at the molecule's position is measured and sent to the "quantum" portion of the code where the density matrix is propagated by the electric field. As an end result, the induced dipole moment of the molecule can be calculated. Finally, the induced dipole moment is fed back into the Meep simulation as the intensity of a point dipole at the position of the molecule.

## Call For Contributions

(as of July 22nd, 2025) This project has been a stepping stone for me in developing my expertise in modern quantum chemistry and computational methods. That being said, when I began work on PlasMol, I had loftier plans than just posting a minimal working version on GitHub, but plans and priorities change. As of the release of v1.0.0, work has paused on this project.

For students in the same or adjacent fields, perhaps the PlasMol skeleton can inspire you to pick it up for your lab's specific desired outcomes. As is the nature of DFT work, one can track many things as a NP + Molecule simulation propagates by contracting the corresponding operator with the current density matrix. Empty commented sections are left in certain files to make adding custom functions easier.

Particular work on monitoring SERS enhancements could put this code to great use, especially given that this was the original intention of the code.

## Citation

There is no publication on this work presently. If I don't get around to getting a publication on this work before you want to use/modify it, please just drop a link in your work to the project's [GitHub](https://github.com/kombatEldridge/PlasMol).

## License

[GPL-3.0 license](https://github.com/kombatEldridge/PlasMol/blob/82712e35aacd37a5e1fb0e5dce74a3f3f678d93f/LICENSE)

## Acknowledgments

- Libraries: [Meep](https://meep.readthedocs.io/en/master/), [PySCF](https://pypi.org/project/pyscf/), [NumPy](https://pypi.org/project/numpy/).
- Contributors: [Brinton Eldridge](https://scholar.google.com/citations?user=8OgnrHMAAAAJ&hl=en&oi=ao).
- Advisors: [Dr. Daniel Nascimento](https://scholar.google.com/citations?user=VVPFNW8AAAAJ&hl=en&oi=ao), [Dr. Yongmei Wang](https://scholar.google.com/citations?user=TLvIKj0AAAAJ&hl=en&oi=ao).

## Contact Information

- Brinton Eldridge
    - Email: [bldrdge1@memphis.edu](mailto:bldrdge1@memphis.edu)
    - GitHub: [https://github.com/kombatEldridge](https://github.com/kombatEldridge?tab=repositories)
    - Organization: University of Memphis
    - LinkedIn: [https://www.linkedin.com/in/brinton-eldridge/](https://www.linkedin.com/in/brinton-eldridge/)
