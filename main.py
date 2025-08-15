from math import radians, degrees, sin, cos, atan2, sqrt, pi

# Constants
mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
seconds_per_day = 86400  # seconds in a day

def orbital_elements_from_TLE(line2):
    """Extract and compute all 7 orbital elements from TLE line 2."""
    parts = line2.split()
    
    # Extract values
    i = float(parts[2])                      # Inclination (degrees)
    Omega = float(parts[3])                  # RAAN (degrees)
    e = float(f"0.{parts[4]}")               # Eccentricity
    omega = float(parts[5])                  # Argument of perigee (degrees)
    M0 = float(parts[6])                     # Mean anomaly (degrees)
    n_rev_per_day = float(parts[7])          # Mean motion (rev/day)

    # Compute semi-major axis using mean motion
    n_rad_per_sec = n_rev_per_day * (2 * pi) / seconds_per_day
    a = (mu / n_rad_per_sec**2) ** (1/3)

    # Solve Kepler’s equation (M = E - e sin E) for E using Newton-Raphson
    M0_rad = radians(M0)
    E = M0_rad  # Initial guess
    while True:
        delta = E - e * sin(E) - M0_rad
        if abs(delta) < 1e-8:
            break
        E -= delta / (1 - e * cos(E))
    
    # Compute true anomaly (θ)
    theta = 2 * atan2(sqrt(1 + e) * sin(E/2), sqrt(1 - e) * cos(E/2))
    theta_deg = degrees(theta) % 360

    return {
        'Semi-major axis a (km)': a,
        'Eccentricity e': e,
        'Inclination i (deg)': i,
        'RAAN Ω (deg)': Omega,
        'Argument of Perigee ω (deg)': omega,
        'Mean Anomaly M0 (deg)': M0,
        'True Anomaly θ (deg)': theta_deg
    }

# TLE Line 2 data
tle_iss = "2 25544  51.6388 282.2982 0005238  28.4654  13.1067 15.49422889504664"
tle_molniya = "2 18946  62.3252 326.4363 7460030 261.3373  16.1061  2.00663207271676"

# Compute and print orbital elements
for name, tle in [("ISS (ZARYA)", tle_iss), ("Molniya 1-71", tle_molniya)]:
    print(f"\n{name} Orbital Elements:")
    elements = orbital_elements_from_TLE(tle)
    for k, v in elements.items():
        print(f"  {k}: {v:.6f}")



from math import radians, degrees, sin, cos, acos, atan2, sqrt, pi

mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
seconds_per_day = 86400

def vector_norm(v):
    return sqrt(sum(x**2 for x in v))

def vector_dot(a, b):
    return sum(i*j for i, j in zip(a, b))

def vector_cross(a, b):
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]

from math import sqrt, acos, atan2, sin, cos, degrees, radians, pi
import numpy as np

mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)

def orbital_elements(r, v):
    r = np.array(r)
    v = np.array(v)

    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    h_hat = h / h_mag

    a = 1 / (2 / r_mag - v_mag ** 2 / mu)

    e_vec = (np.cross(v, h) / mu) - r / r_mag
    e = np.linalg.norm(e_vec)

    i = acos(h_hat[2])

    K = np.array([0, 0, 1])
    N = np.cross(K, h)
    N_mag = np.linalg.norm(N)

    if N_mag < 1e-10:
        Omega = 0
    else:
        N_hat = N / N_mag
        Omega = atan2(N_hat[1], N_hat[0]) % (2 * pi)

    if e < 1e-10:
        omega = 0
    else:
        e_hat = e_vec / e
        if N_mag < 1e-10:
            omega = atan2(e_hat[1], e_hat[0]) % (2 * pi)
        else:
            omega = atan2(np.dot(np.cross(N_hat, e_hat), h_hat), np.dot(N_hat, e_hat)) % (2 * pi)

    if e < 1e-10:
        theta = atan2(np.dot(np.cross(h_hat, r), r), np.dot(h_hat, r)) % (2 * pi)
    else:
        e_hat = e_vec / e
        theta = atan2(np.dot(np.cross(e_hat, r), h_hat), np.dot(e_hat, r)) % (2 * pi)

    if e < 1e-4:
        E = theta
    else:
        E = 2 * atan2(sqrt(1 - e) * sin(theta / 2), sqrt(1 + e) * cos(theta / 2))
    M0 = (E - e * sin(E)) % (2 * pi)

    return {
        'a': a,
        'e': e,
        'i': degrees(i),
        'Omega': degrees(Omega),
        'omega': degrees(omega),
        'M0': degrees(M0),
        'theta': degrees(theta)
    }


def orbital_elements_from_TLE(line2):
    parts = line2.split()
    i = float(parts[2])
    Omega = float(parts[3])
    e = float(f"0.{parts[4]}")
    omega = float(parts[5])
    M0 = float(parts[6])
    n_rev_per_day = float(parts[7])

    n_rad_per_sec = n_rev_per_day * (2 * pi) / seconds_per_day
    a = (mu / n_rad_per_sec**2) ** (1/3)

    M0_rad = radians(M0)
    E = M0_rad
    while True:
        delta = E - e * sin(E) - M0_rad
        if abs(delta) < 1e-8:
            break
        E -= delta / (1 - e * cos(E))

    theta = 2 * atan2(sqrt(1 + e) * sin(E/2), sqrt(1 - e) * cos(E/2))
    return {
        'a': a,
        'e': e,
        'i': i,
        'Omega': Omega,
        'omega': omega,
        'M0': M0,
        'theta': degrees(theta) % 360
    }

def compare_elements(name, from_tle, from_rv):
    print(f"\n{name} - Comparison of Orbital Elements")
    print(f"{'Element':<30}{'From TLE':>15}{'From r,v':>15}{'Difference':>15}")
    print("-" * 75)
    for key in from_tle:
        tle_val = from_tle[key]
        rv_val = from_rv[key]
        diff = tle_val - rv_val
        print(f"{key:<30}{tle_val:15.6f}{rv_val:15.6f}{diff:15.6f}")

# Input data
tle_iss = "2 25544  51.6388 282.2982 0005238  28.4654  13.1067 15.49422889504664"
tle_molniya = "2 18946  62.3252 326.4363 7460030 261.3373  16.1061  2.00663207271676"

r_iss = [3816.2953, -4368.3525, 3535.4439]
v_iss = [2.3924, 5.7259, 4.4942]

r_molniya = [11014.4011, -7331.0311, 36.7577]
v_molniya = [4.9168, -0.3836, 4.5736]

# Calculate orbital elements
iss_tle_elements = orbital_elements_from_TLE(tle_iss)
iss_rv_elements = orbital_elements(r_iss, v_iss)

molniya_tle_elements = orbital_elements_from_TLE(tle_molniya)
molniya_rv_elements = orbital_elements(r_molniya, v_molniya)

# Print comparison tables
compare_elements("ISS (ZARYA)", iss_tle_elements, iss_rv_elements)
compare_elements("Molniya 1-71", molniya_tle_elements, molniya_rv_elements)






import numpy as np

def eci_to_perifocal(r_eci, v_eci, omega, Omega, i):
    """
    Convert position and velocity from ECI to Perifocal frame
    using the 3-1-3 Euler rotation sequence
    """
    # Compute rotation matrix elements
    cO, sO = np.cos(Omega), np.sin(Omega)
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)
    
    # Rotation matrix from perifocal to ECI
    R = np.array([
        [cO*co - sO*so*ci, -cO*so - sO*co*ci,  sO*si],
        [sO*co + cO*so*ci, -sO*so + cO*co*ci, -cO*si],
        [so*si,             co*si,             ci]
    ])
    
    # Transform to Perifocal frame
    r_peri = R.T @ r_eci
    v_peri = R.T @ v_eci
    
    return r_peri, v_peri

# ISS orbital elements in radians
omega_iss = np.radians(28.4654)
Omega_iss = np.radians(282.2982)
i_iss = np.radians(51.6388)

# Molniya orbital elements in radians
omega_mol = np.radians(261.3373)
Omega_mol = np.radians(326.4363)
i_mol = np.radians(62.3252)

# Given initial ECI vectors
r_iss = np.array([3816.2953, -4368.3525, 3535.4439])  # km
v_iss = np.array([2.3924, 5.7259, 4.4942])            # km/s

r_mol = np.array([11014.4011, -7331.0311, 36.7577])   # km
v_mol = np.array([4.9168, -0.3836, 4.5736])           # km/s

# Convert to Perifocal frame
r_iss_peri, v_iss_peri = eci_to_perifocal(r_iss, v_iss, omega_iss, Omega_iss, i_iss)
r_mol_peri, v_mol_peri = eci_to_perifocal(r_mol, v_mol, omega_mol, Omega_mol, i_mol)

# Print results
print("📡 ISS in Perifocal Frame:")
print(f"  Position (km): [{r_iss_peri[0]:.4f}, {r_iss_peri[1]:.4f}, {r_iss_peri[2]:.4f}]")
print(f"  Velocity (km/s): [{v_iss_peri[0]:.4f}, {v_iss_peri[1]:.4f}, {v_iss_peri[2]:.4f}]\n")

print("🛰️ Molniya in Perifocal Frame:")
print(f"  Position (km): [{r_mol_peri[0]:.4f}, {r_mol_peri[1]:.4f}, {r_mol_peri[2]:.4f}]")
print(f"  Velocity (km/s): [{v_mol_peri[0]:.4f}, {v_mol_peri[1]:.4f}, {v_mol_peri[2]:.4f}]")







import numpy as np
import matplotlib.pyplot as plt

# Constants
mu = 398600.4418  # km^3/s^2
T = 24 * 3600     # total time (s)
dt = 10           # time step (s)

def two_body_euler(r0, v0, T, dt):
    """Propagate orbit using Euler integration (friend's algorithm)."""
    t = np.arange(0, T, dt)
    positions = np.zeros((len(t), 3))
    r = r0.copy()
    v = v0.copy()
    
    for i in range(len(t)):
        positions[i] = r
        r_mag = np.linalg.norm(r)
        acceleration = -mu * r / r_mag**3
        v += acceleration * dt
        r += v * dt
    
    return t, positions

# === Initial conditions for ISS and Molniya ===
r0_iss = np.array([3816.2953, -4368.3525, 3535.4439])  # km
v0_iss = np.array([2.3924, 5.7259, 4.4942])            # km/s

r0_mol = np.array([11014.4011, -7331.0311, 36.7577])   # km
v0_mol = np.array([4.9168, -0.3836, 4.5736])           # km/s

# === Propagate and plot ===
for name, r0, v0 in [("ISS", r0_iss, v0_iss), ("Molniya", r0_mol, v0_mol)]:
    t, r = two_body_euler(r0, v0, T, dt)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[:, 0], r[:, 1], r[:, 2], label=f'{name} Trajectory')
    ax.scatter(r0[0], r0[1], r0[2], c='red', label='Initial Position', s=30)
    ax.set_title(f"{name} Trajectory in ECI Frame (24 hours)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()






import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Constants ===
mu = 398600.4418  # km^3/s^2
T = 24 * 3600     # Total propagation time (24 hours)
dt = 10           # Step size in seconds

# === Two-Body Derivatives ===
def two_body_deriv(y):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    a = -mu * r / r_norm**3
    return np.concatenate((v, a))

# === RK4 Integrator ===
def rk4(y0, t0, tf, dt):
    t = np.arange(t0, tf + dt, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = dt * two_body_deriv(y[i-1])
        k2 = dt * two_body_deriv(y[i-1] + 0.5 * k1)
        k3 = dt * two_body_deriv(y[i-1] + 0.5 * k2)
        k4 = dt * two_body_deriv(y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    return t, y

# === Orbit Propagation using solve_ivp ===
def solve_ivp_orbit(y0, t0, tf, dt):
    t_eval = np.arange(t0, tf + dt, dt)
    sol = solve_ivp(lambda t, y: two_body_deriv(y), (t0, tf), y0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T

# === Initial Conditions ===
r0_iss = np.array([3816.2953, -4368.3525, 3535.4439])
v0_iss = np.array([2.3924, 5.7259, 4.4942])
y0_iss = np.concatenate((r0_iss, v0_iss))

r0_mol = np.array([11014.4011, -7331.0311, 36.7577])
v0_mol = np.array([4.9168, -0.3836, 4.5736])
y0_mol = np.concatenate((r0_mol, v0_mol))

# === Run Both Methods for Each Satellite ===
for name, y0 in [("ISS", y0_iss), ("Molniya", y0_mol)]:
    # RK4
    t_rk4, y_rk4 = rk4(y0, 0, T, dt)
    r_rk4 = y_rk4[:, :3]

    # solve_ivp
    t_sci, y_sci = solve_ivp_orbit(y0, 0, T, dt)
    r_sci = y_sci[:, :3]

    # === Plot 3D Trajectory Comparison ===
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_rk4[:, 0], r_rk4[:, 1], r_rk4[:, 2], label='RK4 Trajectory')
    ax.plot(r_sci[:, 0], r_sci[:, 1], r_sci[:, 2], '--', label='solve_ivp Trajectory')
    ax.scatter(y0[0], y0[1], y0[2], color='red', s=40, label='Initial Position')
    ax.set_title(f"{name} Orbit: RK4 vs solve_ivp (ECI Frame, 24h)")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    # === Plot Error Over Time ===
    position_error = np.linalg.norm(r_rk4 - r_sci, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(t_rk4 / 3600, position_error)
    plt.xlabel("Time (hours)")
    plt.ylabel("Position Error (km)")
    plt.title(f"{name} - RK4 vs solve_ivp Position Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()










import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === Constants ===
MU = 398600.4418  # km^3/s^2

# === Orbital Dynamics ===
def acceleration(state, t):
    pos = state[:3]
    vel = state[3:]
    r_mag = np.linalg.norm(pos)
    acc = -MU * pos / r_mag**3
    return np.concatenate((vel, acc))

def rk4_integrate_step(f, current_state, t, h):
    k1 = f(current_state, t)
    k2 = f(current_state + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(current_state + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(current_state + h * k3, t + h)
    return current_state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# === Time & Rotation Calculations ===
def parse_epoch(epoch_str):
    year = 2000 + int(epoch_str[:2])
    doy = float(epoch_str[2:])
    base = datetime(year, 1, 1)
    return base + timedelta(days=doy - 1)

def time_since_j2000(dt):
    y, m, d = dt.year, dt.month, dt.day
    hr, mn, sec = dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
    term1 = 367 * y - np.floor(7 * (y + np.floor((m + 9) / 12)) / 4)
    term2 = np.floor(275 * m / 9)
    day_frac = d + (hr + mn/60 + sec/3600) / 24
    return term1 + term2 + day_frac - 730531.5

def greenwich_sidereal_angle(days_since_2000):
    theta_deg = 280.46061837 + 360.9856473 * days_since_2000
    return np.radians(theta_deg % 360)

def convert_eci_to_ecef(r_eci, t_seconds, epoch_dt):
    dt = epoch_dt + timedelta(seconds=t_seconds)
    d2000 = time_since_j2000(dt)
    theta = greenwich_sidereal_angle(d2000)
    R = np.array([
        [np.cos(theta),  np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R @ r_eci

def ecef_to_latlon(r_ecef):
    x, y, z = r_ecef
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    return np.degrees(lat), np.degrees(lon)

# === Propagation & Ground Track Plotting ===
def simulate_ground_track(r_init, v_init, label, epoch_str, sim_time=86400, step=10):
    epoch = parse_epoch(epoch_str)
    num_steps = int(sim_time / step)
    
    trajectory = np.zeros((num_steps, 6))
    trajectory[0] = np.concatenate((r_init, v_init))
    
    for i in range(1, num_steps):
        trajectory[i] = rk4_integrate_step(acceleration, trajectory[i-1], i*step, step)

    lats, lons = [], []
    for j in range(num_steps):
        r_eci = trajectory[j, :3]
        r_ecef = convert_eci_to_ecef(r_eci, j*step, epoch)
        lat, lon = ecef_to_latlon(r_ecef)
        lats.append(lat)
        lons.append(lon)

    # === Plot ground track ===
    plt.figure(figsize=(10, 5))
    plt.plot(lons, lats, '.', markersize=0.8, label=f'{label} Track')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.title(f'{label} Ground Track (24 hrs)')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Inputs from your assignment ===

epoch_iss = "25100.49497878"      # 2025 Day 100
epoch_mol = "25099.99526772"      # 2025 Day 99

r_iss_eci = np.array([3816.2953, -4368.3525, 3535.4439])
v_iss_eci = np.array([2.3924, 5.7259, 4.4942])

r_mol_eci = np.array([11014.4011, -7331.0311, 36.7577])
v_mol_eci = np.array([4.9168, -0.3836, 4.5736])

# === Run for ISS and Molniya ===
simulate_ground_track(r_iss_eci, v_iss_eci, "ISS", epoch_iss)
simulate_ground_track(r_mol_eci, v_mol_eci, "Molniya", epoch_mol)

