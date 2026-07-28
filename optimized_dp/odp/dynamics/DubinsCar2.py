import heterocl as hcl
import numpy as np

""" 3D DUBINS CAR DYNAMICS with 2 CONTROL INPUTS IMPLEMENTATION 
 x_dot = v * cos(theta)
 y_dot = v * sin(theta)
 theta_dot = w
 u[0] = speed
 u[1] = w
 """


class DubinsCar2:
    def __init__(self, x=[0,0,0], uMin=[-0.1, -1.0], uMax =[0.8, 1.0], dMax=[0,0,0], uMode="min", dMode="max"):
        self.x = x
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode
        self.speedMin = uMin[0]
        self.speedMax = uMax[0]
        self.wMin = uMin[1]
        self.wMax = uMax[1]

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        opt_speed = hcl.scalar(self.speedMax, "opt_speed")
        # Just create and pass back, even though they're not used
        in4 = hcl.scalar(0, "in4")
        # Declare hcl scalars for the coefficient
        deriv0 = hcl.scalar(0, "deriv0")
        deriv1 = hcl.scalar(0, "deriv1")
        theta = hcl.scalar(0, "theta")
        deriv0[0] = spat_deriv[0]
        deriv1[0] = spat_deriv[1]
        theta[0] = state[2]
        # coefficient = spat_deriv[0]*np.cos(state[2]) + spat_deriv[1]*np.sin(state[2])
        coefficient = deriv0[0]*hcl.cos(theta[0]) + deriv1[0]*hcl.sin(theta[0])
    
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(coefficient > 0):
                opt_speed[0] = self.speedMin
            with hcl.if_(spat_deriv[2] > 0):
                opt_w[0] = self.wMin
        with hcl.if_(self.uMode == "max"):
            with hcl.if_(coefficient < 0):
                opt_speed[0] = self.speedMin
            with hcl.elif_(spat_deriv[2] < 0):
                opt_w[0] = self.wMin
            
        return (opt_speed[0], opt_w[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        """
        Bang-bang optimal disturbance over [-dMax, +dMax] in each axis.
        dMode == "max": align sign with spat_deriv (maximize Hamiltonian).
        dMode == "min": oppose sign of spat_deriv (minimize Hamiltonian).
        """
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")

        with hcl.if_(self.dMode == "max"):
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = self.dMax[0]
            with hcl.else_():
                d1[0] = -self.dMax[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = self.dMax[1]
            with hcl.else_():
                d2[0] = -self.dMax[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = self.dMax[2]
            with hcl.else_():
                d3[0] = -self.dMax[2]
        with hcl.else_():
            with hcl.if_(spat_deriv[0] >= 0):
                d1[0] = -self.dMax[0]
            with hcl.else_():
                d1[0] = self.dMax[0]
            with hcl.if_(spat_deriv[1] >= 0):
                d2[0] = -self.dMax[1]
            with hcl.else_():
                d2[0] = self.dMax[1]
            with hcl.if_(spat_deriv[2] >= 0):
                d3[0] = -self.dMax[2]
            with hcl.else_():
                d3[0] = self.dMax[2]

        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = uOpt[0]*hcl.cos(state[2]) + dOpt[0]
        y_dot[0] = uOpt[0]*hcl.sin(state[2]) + dOpt[1]
        theta_dot[0] = uOpt[1] + dOpt[2]

        return (x_dot[0], y_dot[0], theta_dot[0])
    
    def optCtrl_inPython(self, state, spat_deriv):
        opt_w = self.wMax
        opt_speed = self.speedMax
        coefficient = spat_deriv[0]*np.cos(state[2]) + spat_deriv[1]*np.sin(state[2])
        
        if self.uMode == "min":
            if spat_deriv[2] > 0:
                opt_w = self.wMin
            if coefficient > 0:
                opt_speed = self.speedMin
        else:
            if spat_deriv[2] < 0:
                opt_w = self.wMin
            if coefficient < 0:
                opt_speed = self.speedMin
        # if spat_deriv[2] > 0:
        #     if self.uMode == "min":
        #         opt_w = - self.wMax
        # else:
        #     if self.uMode == "max":
        #         opt_w = - self.wMax
        # print(opt_speed, opt_w)
        return np.array([opt_speed, opt_w])
    
    def optDstb_inPython(self, state, spat_deriv):
        """
        Bang-bang optimal disturbance for Dubins Car.
        
        Since dynamics are:
            x_dot = v*cos(theta) + d1
            y_dot = v*sin(theta) + d2
            theta_dot = w + d3
        
        dMode="max": pick dMax if p_i >= 0, else dMin (maximise H)
        dMode="min": pick dMin if p_i >= 0, else dMax (minimise H)
        """
        if self.dMode == "max":
            opt_d1 = self.dMax[0] if spat_deriv[0] >= 0 else -self.dMax[0]
            opt_d2 = self.dMax[1] if spat_deriv[1] >= 0 else -self.dMax[1]
            opt_d3 = self.dMax[2] if spat_deriv[2] >= 0 else -self.dMax[2]
        else:  # "min"
            opt_d1 = -self.dMax[0] if spat_deriv[0] >= 0 else self.dMax[0]
            opt_d2 = -self.dMax[1] if spat_deriv[1] >= 0 else self.dMax[1]
            opt_d3 = -self.dMax[2] if spat_deriv[2] >= 0 else self.dMax[2]

        return np.array([opt_d1, opt_d2, opt_d3])
    
    def dynamics_inPython(self, state, control, disturbance=None):
        """Return the partial derivative equations of one agent.

        Args:
            state (np.ndarray, shape(3, )): the state of one agent
            action (np.ndarray, shape (1, )): the action of one agent
        """

        if disturbance is None:
            disturbance = np.zeros(3)
        dx = control[0] * np.cos(state[2]) + disturbance[0]
        dy = control[0] * np.sin(state[2]) + disturbance[1]
        dtheta = control[1] + disturbance[2]
        return (dx, dy, dtheta)
    
    # =========================
    # tTLT / safety-filter abstraction layer
    # =========================
    # These four methods expose a dynamics-agnostic interface used by
    # Backtracking / is_valid_control in tTLT_synthesis.py.  Same shape
    # as Plane2D's so the safety filter is dynamics-agnostic.

    def control_box(self) -> tuple:
        """Control input box  ((u1_min, u1_max), (u2_min, u2_max))
        with u1 = speed and u2 = angular rate ω."""
        return ((self.speedMin, self.speedMax), (self.wMin, self.wMax))

    def control_labels(self) -> tuple:
        """Human-readable names for the two control coordinates."""
        return ("v", "ω")

    def control_halfplane_coeffs(self, state, spat_deriv) -> tuple:
        """
        Coefficients (α, β) of the safety half-plane  α·v + β·ω + γ ≤ 0
        in control space, derived from V̇ = ∇V · ẋ for Dubins:

            V̇ = (P_x cos θ + P_y sin θ)·v  +  P_θ·ω  +  (P_x d_x + P_y d_y + P_θ d_θ)
                ╰────────────── α ──────────╯  ╰─ β ─╯  ╰─────────── γ ───────────╯

        State must include θ as state[2].
        """
        theta = float(state[2])
        alpha = float(spat_deriv[0]) * np.cos(theta) + float(spat_deriv[1]) * np.sin(theta)
        beta  = float(spat_deriv[2])
        return (alpha, beta)

    def disturbance_offset(self, state, spat_deriv) -> tuple:
        """
        Worst-case disturbance contribution to V̇:
            γ       = max_{d ∈ D}(P_x d_x + P_y d_y + P_θ d_θ) = ∇V · d_worst
            d_worst = argmax_d (∇V · d)   (bang-bang under box D = [-dMax, +dMax]³)

        Returns (γ, d_worst) where d_worst is a numpy array of shape (3,).
        """
        d_worst = self.optDstb_inPython(state, spat_deriv)
        gamma   = float(np.dot(np.asarray(spat_deriv, dtype=float), d_worst))
        return gamma, d_worst

    def forward(self, ctrl_freq, current_state, control):
        # Forward the dubincar dynamics with one step
        x, y, theta = current_state
        dt = 1.0 / ctrl_freq
        
        # Forward-Euler method
        next_x = x + control[0] * np.cos(theta) * dt
        next_y = y + control[0] * np.sin(theta) * dt
        next_theta_raw = theta + control[1] * dt

        def check_theta(angle):
            # Make sure the angle is in the range of [0, 2*pi)
            while angle >=2*np.pi:
                angle -= 2 * np.pi
            while angle < 0:
                angle += 2 * np.pi

            return angle

        # Check the boundary
        next_theta = check_theta(next_theta_raw)
        next_state = (next_x, next_y, next_theta)
        
        return next_state
        
        