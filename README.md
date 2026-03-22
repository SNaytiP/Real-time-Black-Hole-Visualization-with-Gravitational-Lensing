# Real-time Black Hole Simulation with Gravitational Lensing

> A real-time interactive simulation of black hole physics, combining professional astronomical computing libraries with GPU-accelerated rendering techniques.

---

## Abstract

This project presents a real-time simulation of black hole phenomena, implemented as a final project for a Real-time Computer Graphics course. The simulation integrates **REBOUND**, an N-body integrator used in peer-reviewed astrophysics research, with OpenGL-based rendering to produce physically grounded and visually accurate representations of black hole environments. Key phenomena modelled include gravitational lensing, accretion disk dynamics, and orbital mechanics under general relativistic corrections.

---

## Features

- **Gravitational Lensing** — Screen-space GLSL fragment shader approximating photon path deflection around a Schwarzschild black hole
- **Accretion Disk Simulation** — GPU particle system with angular momentum conservation and Doppler color shift (relativistic blueshift / redshift)
- **N-body Orbital Mechanics** — Powered by REBOUND with REBOUNDx general relativity extensions (frame-dragging, perihelion precession)
- **Procedural Starfield & Nebula** — Perlin/simplex noise-based background with HDR bloom post-processing
- **Interactive Events** — Real-time tidal disruption event triggered by user input; adjustable black hole mass via UI

---

## Physics Background

### Schwarzschild Metric & Gravitational Lensing

The gravitational lensing effect is based on the Schwarzschild solution to Einstein's field equations, describing the spacetime geometry around a non-rotating, spherically symmetric mass:

$$ds^2 = -\left(1 - \frac{r_s}{r}\right)c^2\,dt^2 + \left(1 - \frac{r_s}{r}\right)^{-1}dr^2 + r^2\,d\Omega^2$$

where the Schwarzschild radius is defined as:

$$r_s = \frac{2GM}{c^2}$$

In the real-time renderer, we approximate the photon deflection angle using the weak-field lensing formula:

$$\hat{\alpha} = \frac{4GM}{c^2 b} = \frac{2r_s}{b}$$

where $b$ is the impact parameter (closest approach distance of the light ray). This is implemented as a UV displacement map in the GLSL fragment shader.

### Accretion Disk & Doppler Shift

The accretion disk is modelled as a particle system governed by Keplerian orbital velocity:

$$v_{\text{orb}} = \sqrt{\frac{GM}{r}}$$

Relativistic Doppler color shift is applied per-particle based on the line-of-sight velocity component. Particles approaching the observer are shifted toward blue ($\lambda' < \lambda$), while receding particles shift toward red:

$$\frac{\lambda_{\text{obs}}}{\lambda_{\text{emit}}} = \sqrt{\frac{1 - \beta}{1 + \beta}}, \quad \beta = \frac{v}{c}$$

### N-body Integration (REBOUND)

Orbital mechanics are computed using REBOUND's IAS15 integrator — a 15th-order adaptive integrator with near-machine-precision error control, used in published astrophysics literature. General relativistic corrections (post-Newtonian 1PN) are applied via REBOUNDx.

---

## Technology Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| N-body physics | [REBOUND](https://rebound.readthedocs.io/) | Orbital integration, collision detection |
| GR corrections | [REBOUNDx](https://reboundx.readthedocs.io/) | Post-Newtonian forces, frame-dragging |
| Astronomical units | [Astropy](https://www.astropy.org/) | Physical constants, unit conversion |
| Real-time rendering | OpenGL + GLSL | Shader pipeline, particle rendering |
| GPU visualization | [Vispy](https://vispy.org/) | High-performance Python–OpenGL bridge |
| Post-processing | Custom GLSL | Bloom, HDR tone-mapping, lensing shader |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/black-hole-simulation.git
cd black-hole-simulation

# Install dependencies
pip install rebound reboundx astropy vispy numpy

# Run the simulation
python main.py
```

> Requires Python 3.9+, OpenGL 3.3+, and a GLSL-compatible GPU.

---

## References

1. Rein, H. & Liu, S.-F. (2012). *REBOUND: An open-source multi-purpose N-body code for collisional dynamics*. Astronomy & Astrophysics, 537, A128.
2. Tamayo, D. et al. (2020). *REBOUNDx: A library for adding conservative and dissipative forces to otherwise symplectic N-body integrations*. MNRAS, 491(2), 2885–2901.
3. James, O. et al. (2015). *Gravitational lensing by spinning black holes in astrophysics, and in the movie Interstellar*. Classical and Quantum Gravity, 32(6), 065001.
4. Astropy Collaboration (2022). *The Astropy Project: Sustaining and growing a community-developed open-source project and status of the v5.0 core package*. ApJ, 935, 167.

---

## Course Information

This repository contains the final project submission for:

**2110514 — Real-time Computer Graphics and Physics Simulation**
Semester 2 · Department of Computer Engineering, Chulalongkorn University

> Submitted as part of the Final Project Proposal and implementation requirement for 2110514.

---

## License

This project is released under the [MIT License](LICENSE).
