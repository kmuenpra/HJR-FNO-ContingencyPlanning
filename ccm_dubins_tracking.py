import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Dubins dynamics
# ===============================

def dubins_step(x, omega, v, dt):
    """
    One integration step of Dubins vehicle.
    x = [x, y, theta]
    """
    xn = np.zeros_like(x)
    xn[0] = x[0] + v * np.cos(x[2]) * dt
    xn[1] = x[1] + v * np.sin(x[2]) * dt
    xn[2] = x[2] + omega * dt
    return xn


# ===============================
# 2. CCM feedback controller
# ===============================

def ccm_feedback(x, xd, omega_d, W, lam):
    """
    CCM tracking controller (geodesic approx).
    """
    dx = x - xd

    # wrap angle error
    dx[2] = np.arctan2(np.sin(dx[2]), np.cos(dx[2]))

    a, b, c, d = W["a"], W["b"], W["c"], W["d"]

    # full CCM feedback (general form)
    omega = omega_d - (lam / c) * (
        d * np.cos(x[2]) * dx[0]
        + d * np.sin(x[2]) * dx[1]
        + c * dx[2]
    )

    return omega


# ===============================
# 3. Reference trajectory
# ===============================

def reference(t, R=5.0, omega0=0.2):
    xd = np.zeros(3)
    xd[0] = R * np.cos(omega0 * t)
    xd[1] = R * np.sin(omega0 * t)
    xd[2] = omega0 * t + np.pi / 2
    omega_d = omega0
    return xd, omega_d


# ===============================
# 4. Simulation
# ===============================

dt = 0.01
T = 50.0
N = int(T / dt)
v = 1.0
lam = 1

# CCM parameters (from solver)
W = {
'a' : 0.10100450092495634 ,
'b' : 0.00010000161531987283 ,
'c' : 0.10099549729509211 ,
    'd' : 0.05
}

# initial condition (off the trajectory)
x = np.array([6.0, 0.0, 0.0])

traj = []
traj_ref = []

for k in range(N):
    t = k * dt
    xd, omega_d = reference(t)

    omega = ccm_feedback(x, xd, omega_d, W, lam)
    x = dubins_step(x, omega, v, dt)

    traj.append(x.copy())
    traj_ref.append(xd.copy())

traj = np.array(traj)
traj_ref = np.array(traj_ref)

# ===============================
# 5. Plot
# ===============================

plt.figure(figsize=(6,6))
plt.plot(traj[:,0], traj[:,1], label="Dubins + CCM")
plt.plot(traj_ref[:,0], traj_ref[:,1], "--", label="Reference")
plt.axis("equal")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Dubins trajectory tracking with CCM feedback")
plt.show()
