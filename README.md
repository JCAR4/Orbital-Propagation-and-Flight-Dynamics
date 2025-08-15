# Orbital Propagation and Flight Dynamics

Python toolkit (AS3050 – Flight Dynamics 2, IIT Madras) for extracting orbital elements, propagating satellite trajectories, and plotting ground tracks (ISS & Molniya).

## Features
- COEs from **TLE** and from **ECI state vectors (r, v)**; side-by-side comparison.
- **ECI ↔ Perifocal** transforms (3-1-3 Euler sequence).
- Orbit propagation: **Euler**, **RK4**, and **SciPy `solve_ivp` (DOP853)**.
- **Error growth** analysis over 24 h against `solve_ivp`.
- **Ground tracks**: ECI → ECEF (GST) → lat/long → 2D map.

## Install
```bash
pip install -r requirements.txt
