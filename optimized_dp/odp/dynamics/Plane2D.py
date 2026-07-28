import heterocl as hcl
import numpy as np


class Plane2D:
    """
    2D Single Integrator with Additive Disturbance

    Dynamics:
        x_dot = u_x + d_x
        y_dot = u_y + d_y

    Hamiltonian:
        H(x, p, u, d) = p_x * (u_x + d_x) + p_y * (u_y + d_y)

    where p = ∇V is the spatial gradient of the value function.

    Optimal control (bang-bang, separable in each dimension):
        uMode="max"  →  maximise H  →  u_x = vxMax if p_x >= 0, else vxMin
        uMode="min"  →  minimise H  →  u_x = vxMin if p_x >= 0, else vxMax

    Optimal disturbance (bang-bang, separable in each dimension):
        dMode="max"  →  maximise H  →  d_x = dMax if p_x >= 0, else dMin
        dMode="min"  →  minimise H  →  d_x = dMin if p_x >= 0, else dMax

    Control bounds:
        u_x ∈ [vxMin, vxMax],  u_y ∈ [vyMin, vyMax]

    Disturbance bounds:
        d_x, d_y ∈ [dMin, dMax]
    """

    def __init__(self,
                 x=[0, 0],
                 vxMin=-1, vxMax=1,
                 vyMin=-1, vyMax=1,
                 dMin=0.0, dMax=0.0,
                 uMode="min",
                 dMode="max"):

        self.x = x

        self.vxMin = vxMin
        self.vxMax = vxMax
        self.vyMin = vyMin
        self.vyMax = vyMax

        self.dMin = dMin
        self.dMax = dMax

        self.uMode = uMode
        self.dMode = dMode

    # =========================
    # Optimal Control
    # =========================
    def opt_ctrl(self, t, state, spat_deriv):
        """
        Bang-bang optimal control.

        uMode="max": opt_vx = vxMax if p_x >= 0 else vxMin  (maximise H)
        uMode="min": opt_vx = vxMin if p_x >= 0 else vxMax  (minimise H)

        Default initialised to vxMax / vyMax (the uMode="max", p>=0 case).
        """
        opt_vx = hcl.scalar(self.vxMax, "opt_vx")
        opt_vy = hcl.scalar(self.vyMax, "opt_vy")

        # x-direction
        with hcl.if_(spat_deriv[0] >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_vx[0] = self.vxMin
        with hcl.else_():                       # p_x < 0
            with hcl.if_(self.uMode == "max"):
                opt_vx[0] = self.vxMin

        # y-direction
        with hcl.if_(spat_deriv[1] >= 0):
            with hcl.if_(self.uMode == "min"):
                opt_vy[0] = self.vyMin
        with hcl.else_():                       # p_y < 0
            with hcl.if_(self.uMode == "max"):
                opt_vy[0] = self.vyMin

        return (opt_vx[0], opt_vy[0])

    # =========================
    # Optimal Disturbance
    # =========================
    def opt_dstb(self, t, state, spat_deriv):
        """
        Bang-bang optimal disturbance.

        dMode="max": d_x = dMax if p_x >= 0 else dMin  (maximise H)
        dMode="min": d_x = dMin if p_x >= 0 else dMax  (minimise H)

        Default initialised to dMax (the dMode="max", p>=0 case).
        """
        d1 = hcl.scalar(self.dMax, "d1")
        d2 = hcl.scalar(self.dMax, "d2")

        # x-direction
        with hcl.if_(spat_deriv[0] >= 0):
            with hcl.if_(self.dMode == "min"):
                d1[0] = self.dMin
        with hcl.else_():                       # p_x < 0
            with hcl.if_(self.dMode == "max"):
                d1[0] = self.dMin

        # y-direction
        with hcl.if_(spat_deriv[1] >= 0):
            with hcl.if_(self.dMode == "min"):
                d2[0] = self.dMin
        with hcl.else_():                       # p_y < 0
            with hcl.if_(self.dMode == "max"):
                d2[0] = self.dMin

        return (d1[0], d2[0])

    # =========================
    # System Dynamics
    # =========================
    def dynamics(self, t, state, uOpt, dOpt):
        """
        x_dot = u_x + d_x
        y_dot = u_y + d_y
        """
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")

        x_dot[0] = uOpt[0] + dOpt[0]
        y_dot[0] = uOpt[1] + dOpt[1]

        return (x_dot[0], y_dot[0])

    # =========================
    # Python / NumPy equivalents
    # =========================
    def optCtrl_inPython(self, state, spat_deriv) -> np.ndarray:
        """
        NumPy equivalent of opt_ctrl.

        Parameters
        ----------
        state      : np.ndarray (2,)  — unused for single integrator
        spat_deriv : tuple | np.ndarray  (p_x, p_y) = ∇V

        Returns
        -------
        np.ndarray [opt_vx, opt_vy]
        """
        if self.uMode == "max":
            opt_vx = self.vxMax if spat_deriv[0] >= 0 else self.vxMin
            opt_vy = self.vyMax if spat_deriv[1] >= 0 else self.vyMin
        elif self.uMode == "min":
            opt_vx = self.vxMin if spat_deriv[0] >= 0 else self.vxMax
            opt_vy = self.vyMin if spat_deriv[1] >= 0 else self.vyMax
        else:
            raise ValueError(f"Unknown uMode: {self.uMode}")

        return np.array([opt_vx, opt_vy])

    def optDstb_inPython(self, state, spat_deriv) -> np.ndarray:
        """
        NumPy equivalent of opt_dstb.

        Parameters
        ----------
        state      : np.ndarray (2,)  — unused for single integrator
        spat_deriv : tuple | np.ndarray  (p_x, p_y) = ∇V

        Returns
        -------
        np.ndarray [opt_dx, opt_dy]
        """
        if self.dMode == "max":
            opt_dx = self.dMax if spat_deriv[0] >= 0 else self.dMin
            opt_dy = self.dMax if spat_deriv[1] >= 0 else self.dMin
        elif self.dMode == "min":
            opt_dx = self.dMin if spat_deriv[0] >= 0 else self.dMax
            opt_dy = self.dMin if spat_deriv[1] >= 0 else self.dMax
        else:
            raise ValueError(f"Unknown dMode: {self.dMode}")

        return np.array([opt_dx, opt_dy])

    def dynamics_inPython(self, state, control, disturbance=None) -> tuple:
        """
        x_dot = u_x + d_x
        y_dot = u_y + d_y

        Parameters
        ----------
        state       : np.ndarray (2,)  [x, y]
        control     : np.ndarray (2,)  [vx, vy]
        disturbance : np.ndarray (2,) | None  [dx, dy]; zeros if None

        Returns
        -------
        tuple (dx, dy)
        """
        if disturbance is None:
            disturbance = np.zeros(2)
        return (control[0] + disturbance[0],
                control[1] + disturbance[1])

    # =========================
    # tTLT / safety-filter abstraction layer
    # =========================
    # These four methods expose a dynamics-agnostic interface used by
    # Backtracking / is_valid_control in tTLT_synthesis.py.  Implementing
    # them on each dynamics class lets the same safety-filter code work
    # for both Plane2D (single integrator) and DubinsCar2.

    def control_box(self) -> tuple:
        """Control input box  ((u1_min, u1_max), (u2_min, u2_max))."""
        return ((self.vxMin, self.vxMax), (self.vyMin, self.vyMax))

    def control_labels(self) -> tuple:
        """Human-readable names for the two control coordinates."""
        return ("v_x", "v_y")

    def control_halfplane_coeffs(self, state, spat_deriv) -> tuple:
        """
        Coefficients (α, β) such that the safety condition  V̇ ≤ 0  becomes
        the half-plane  α·u1 + β·u2 + γ ≤ 0  in (u1, u2) = (v_x, v_y) space.

        For ẋ = u + d (single integrator), V̇ = P_x v_x + P_y v_y + … , so
        (α, β) = (P_x, P_y) — the spatial gradient IS the half-plane normal.
        """
        return (float(spat_deriv[0]), float(spat_deriv[1]))

    def disturbance_offset(self, state, spat_deriv) -> tuple:
        """
        Worst-case disturbance contribution to V̇:
            γ       = max_{d ∈ D}(∇V · d)
            d_worst = argmax_d (∇V · d)   (bang-bang under box D = [dMin,dMax]²)

        Returns (γ, d_worst) where d_worst is a numpy array of shape (state_dim,).
        """
        d_worst = self.optDstb_inPython(state, spat_deriv)
        gamma   = float(np.dot(np.asarray(spat_deriv, dtype=float), d_worst))
        return gamma, d_worst