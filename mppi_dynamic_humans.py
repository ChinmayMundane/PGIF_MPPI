"""
MPPI Planner for Ego Vehicle with Dynamic Humans
=================================================
Cost function uses 2D anisotropic Gaussian fields surrounding each human,
oriented along their direction of motion.

Human cost field:
    C_h(x, y, t) = A * exp( -0.5 * [dx, dy] @ Sigma_inv @ [dx, dy].T )

where Sigma is an anisotropic covariance aligned with the heading direction,
with a larger spread in the forward direction (motion cone).
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, vmap
import jax.lax as lax
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import time

# Copy the proj_mppi class here


class proj_mppi:
    def __init__(self, num_batch):
        self.v_min = 10 ** (-6)
        self.v_max = 2.0
        self.psi_dot_max = 1.5
        self.psi_dot_min = -1.5
        self.t_fin = 4
        self.num = 40
        self.dt = self.t_fin / self.num
        self.radius = 0.5

        self.param_lambda = 1000.0
        self.param_gamma = self.param_lambda * 0.01

        self.P_jax = jnp.identity(self.num)
        self.nvar = jnp.shape(self.P_jax)[0]
        self.num_batch = num_batch

        # Cost weights
        self.w_1 = 1000.0  # Goal reaching (per-step)
        self.w_2 = 1000000  # Static obstacles
        self.w_3 = 0.0  # MPPI regularisation
        self.w_4 = 1000.0  # Terminal goal
        self.w_5 = 5000.0  # Global path tracking
        self.w_6 = 1000000.0  # Human Gaussian field cost
        # Vmapped helpers
        self.v_closest_point_on_segment = jax.vmap(
            self.closest_point_on_segment, in_axes=(None, 0, 0)
        )
        self.compute_cost_mppi_batch = jit(
            vmap(
                self.compute_cost_mppi,
                in_axes=(0, 0, 0, None, None, None, None, None, None, None),
            )
        )
        self.compute_weights_batch = jit(
            vmap(self._compute_weights, in_axes=(0, None, None))
        )
        self.compute_epsilon_batch = jit(vmap(self.compute_epsilon, in_axes=(1, None)))
        self.compute_w_epsilon_batch = jit(vmap(self.compute_w_epsilon, in_axes=(0, 0)))
        self.goal_reaching_batch = jit(
            vmap(self.goal_reaching_cost_single, in_axes=(0, 0, None, None))
        )

    # Control sampling

    @partial(jit, static_argnums=(0,))
    def compute_control_samples(self, key, mean_control, cov_control):
        key, subkey = jax.random.split(key)
        control_samples = jax.random.multivariate_normal(
            key, mean_control, cov_control, (self.num_batch,)
        )
        c_v_samples = control_samples[:, 0 : self.nvar]
        c_psi_samples = control_samples[:, self.nvar : 2 * self.nvar]
        return c_v_samples, c_psi_samples, key

    # Rollouts

    @partial(jit, static_argnums=(0,))
    def compute_rollout_one_step(self, v_samples, psidot, x, y, psi):
        psi_next = psi + psidot * self.dt
        vx_next = v_samples * jnp.cos(psi_next)
        vy_next = v_samples * jnp.sin(psi_next)
        x_next = x + vx_next * self.dt
        y_next = y + vy_next * self.dt
        return x_next, y_next, psi_next

    @partial(jit, static_argnums=(0,))
    def compute_rollouts(self, x_init, y_init, psi_init, v_samples, psidot):
        x_roll_init = jnp.zeros((self.num_batch, self.num)).at[:, 0].set(x_init)
        y_roll_init = jnp.zeros((self.num_batch, self.num)).at[:, 0].set(y_init)
        psi_roll_init = jnp.zeros((self.num_batch, self.num)).at[:, 0].set(psi_init)

        def lax_rollout(carry, idx):
            x_roll, y_roll, psi_roll = carry
            x_next, y_next, psi_next = self.compute_rollout_one_step(
                v_samples[:, idx],
                psidot[:, idx],
                x_roll[:, idx],
                y_roll[:, idx],
                psi_roll[:, idx],
            )
            x_roll = x_roll.at[:, idx + 1].set(x_next)
            y_roll = y_roll.at[:, idx + 1].set(y_next)
            psi_roll = psi_roll.at[:, idx + 1].set(psi_next)
            return (x_roll, y_roll, psi_roll), 0.0

        carry_init = x_roll_init, y_roll_init, psi_roll_init
        carry_final, _ = jax.lax.scan(lax_rollout, carry_init, jnp.arange(self.num - 1))
        return carry_final

    @partial(jit, static_argnums=(0,))
    def compute_rollouts_single(self, x_init, y_init, psi_init, v_samples, psidot):
        x_roll_init = jnp.zeros(self.num).at[0].set(x_init)
        y_roll_init = jnp.zeros(self.num).at[0].set(y_init)
        psi_roll_init = jnp.zeros(self.num).at[0].set(psi_init)

        def lax_rollout(carry, idx):
            x_roll, y_roll, psi_roll = carry
            x_next, y_next, psi_next = self.compute_rollout_one_step(
                v_samples[idx], psidot[idx], x_roll[idx], y_roll[idx], psi_roll[idx]
            )
            x_roll = x_roll.at[idx + 1].set(x_next)
            y_roll = y_roll.at[idx + 1].set(y_next)
            psi_roll = psi_roll.at[idx + 1].set(psi_next)
            return (x_roll, y_roll, psi_roll), 0.0

        carry_init = x_roll_init, y_roll_init, psi_roll_init
        carry_final, _ = jax.lax.scan(lax_rollout, carry_init, jnp.arange(self.num - 1))
        return carry_final

    # Path-tracking

    @partial(jit, static_argnums=(0,))
    def closest_point_on_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_norm_sq = jnp.sum(ab**2)
        ab_norm_sq = jnp.where(ab_norm_sq == 0, 1e-6, ab_norm_sq)
        t = jnp.clip(jnp.dot(ap, ab) / ab_norm_sq, 0.0, 1.0)
        closest = a + t * ab
        dist = jnp.linalg.norm(p - closest)
        tangent = ab / (jnp.linalg.norm(ab) + 1e-6)
        return closest, dist, tangent

    @partial(jit, static_argnums=(0,))
    def compute_perpendicular_errors(self, global_path, trajectories):
        seg_starts = global_path[:-1]
        seg_ends = global_path[1:]
        traj_flat = trajectories.reshape(-1, 2)

        def process_point(p):
            closest_pts, dists, tangents = self.v_closest_point_on_segment(
                p, seg_starts, seg_ends
            )
            min_idx = jnp.argmin(dists)
            tangent = tangents[min_idx]
            normal = jnp.array([-tangent[1], tangent[0]])
            offset = p - closest_pts[min_idx]
            return jnp.abs(jnp.dot(offset, normal))

        lateral_errors = jax.vmap(process_point)(traj_flat)
        lateral_errors = lateral_errors.reshape(self.num_batch, self.num)
        return jnp.linalg.norm(lateral_errors, axis=1)

    # Cost functions

    @partial(jit, static_argnums=(0,))
    def obstacle_cost(self, x_obs, y_obs, x, y):
        obstacle = (x - x_obs) ** 2 + (y - y_obs) ** 2 - self.radius**2
        return jnp.maximum(0.0, -obstacle)

    @partial(jit, static_argnums=(0,))
    def human_gaussian_field(self, ego_x, ego_y, h_x, h_y, h_vx, h_vy):
        """
        2D anisotropic Gaussian field for a single human at one timestep.

        The Gaussian is elongated *ahead* of the human (along their velocity
        direction) and narrow laterally, representing a motion-cone danger zone.

              sigma_along : spread forward/backward (direction of motion)
              sigma_perp  : spread left/right (perpendicular)

        C = A * exp( -0.5 * mahalanobis^2 )
        """
        dx = ego_x - h_x
        dy = ego_y - h_y

        speed = jnp.sqrt(h_vx**2 + h_vy**2) + 1e-6
        # Unit vector along motion
        ux = h_vx / speed
        uy = h_vy / speed
        # Unit vector perpendicular to motion
        nx = -uy
        ny = ux

        # Project displacement onto motion-aligned axes
        d_along = dx * ux + dy * uy
        d_perp = dx * nx + dy * ny

        # Anisotropic spread: wider ahead (d_along > 0), narrower behind
        sigma_perp = 0.6  # lateral sigma [m]
        sigma_along_fwd = 1.2 + 0.5 * speed  # forward sigma (grows with speed)
        sigma_along_bwd = 0.5  # backward sigma (small)

        sigma_along = jnp.where(d_along >= 0, sigma_along_fwd, sigma_along_bwd)

        exponent = 0.5 * ((d_along / sigma_along) ** 2 + (d_perp / sigma_perp) ** 2)
        amplitude = 1.0
        return amplitude * jnp.exp(-exponent)

    @partial(jit, static_argnums=(0,))
    def human_gaussian_cost_rollout(
        self, x_traj, y_traj, h_x_traj, h_y_traj, h_vel_traj
    ):
        """
        Sum Gaussian field over horizon for one sample trajectory vs all humans.
        x_traj, y_traj : (num,)
        h_x_traj       : (num_humans, num)
        h_y_traj       : (num_humans, num)
        h_vel_traj     : (num_humans, num, 2)  [vx, vy]
        """
        if h_x_traj.shape[0] == 0:
            return 0.0

        def cost_at_step(t):
            def cost_per_human(h_idx):
                return self.human_gaussian_field(
                    x_traj[t],
                    y_traj[t],
                    h_x_traj[h_idx, t],
                    h_y_traj[h_idx, t],
                    h_vel_traj[h_idx, t, 0],
                    h_vel_traj[h_idx, t, 1],
                )

            return jnp.sum(jax.vmap(cost_per_human)(jnp.arange(h_x_traj.shape[0])))

        return jnp.sum(jax.vmap(cost_at_step)(jnp.arange(self.num)))

    @partial(jit, static_argnums=(0,))
    def compute_cost_mppi(
        self,
        controls_stack,
        x,
        y,
        x_goal,
        y_goal,
        x_obs,
        y_obs,
        h_x_traj,
        h_y_traj,
        h_vel_traj,
    ):
        """
        Calculates the scalar cost for ONE sample rollout over the entire horizon.
        """
        # Static obstacles (summing over obstacles, for this one sample's horizon)
        cost_obs = (
            jnp.sum(
                jax.vmap(lambda xo, yo: jnp.sum(self.obstacle_cost(xo, yo, x, y)))(
                    x_obs, y_obs
                )
            )
            * self.w_2
        )

        # Human Gaussian field
        cost_human = (
            self.human_gaussian_cost_rollout(x, y, h_x_traj, h_y_traj, h_vel_traj)
            * self.w_6
        )

        # MPPI regularisation
        u_mean = jnp.mean(controls_stack, axis=0)
        mppi_reg = self.param_gamma * jnp.sum(u_mean * controls_stack) * self.w_3

        return cost_obs + cost_human + mppi_reg

    @partial(jit, static_argnums=(0,))
    def compute_epsilon(self, epsilon, w):
        we = self.compute_w_epsilon_batch(epsilon, w)
        return jnp.sum(we, axis=0)

    @partial(jit, static_argnums=(0,))
    def compute_w_epsilon(self, epsilon, w):
        return w * epsilon

    @partial(jit, static_argnums=(0,))
    def _compute_weights(self, S, rho, eta):
        return (1.0 / eta) * jnp.exp((-1.0 / self.param_lambda) * (S - rho))

    @partial(jit, static_argnums=(0,))
    def dot_controls(self, v, psi, v_init, psi_init, v_dot_init, psi_dot_init):
        v_d = jnp.zeros(self.num + 1).at[0].set(v_init).at[1:].set(v)
        v_dot = jnp.diff(v_d) / self.dt
        v_dd = jnp.zeros(self.num + 1).at[0].set(v_dot_init).at[1:].set(v_dot)
        v_ddot = jnp.diff(v_dd) / self.dt
        psi_d = jnp.zeros(self.num + 1).at[0].set(psi_init).at[1:].set(psi)
        psi_dot = jnp.diff(psi_d) / self.dt
        psi_dd = jnp.zeros(self.num + 1).at[0].set(psi_dot_init).at[1:].set(psi_dot)
        psi_ddot = jnp.diff(psi_dd) / self.dt
        return v_dot, v_ddot, psi_dot, psi_ddot

    @partial(jit, static_argnums=(0,))
    def goal_reaching_cost_single(self, traj_x, traj_y, goal_x, goal_y):
        distances = jnp.sqrt((traj_x - goal_x) ** 2 + (traj_y - goal_y) ** 2)
        return jnp.min(distances)

    # ── Main MPPI solve ───────────────────────────────────────────────────────

    @partial(jit, static_argnums=(0,))
    def compute_mppi(
        self,
        v_init,
        v_dot_init,
        psi_init,
        psi_dot_init,
        psi_ddot_init,
        x_init,
        y_init,
        x_fin,
        y_fin,
        mean_control,
        key,
        x_obs,
        y_obs,
        global_path,
        h_x_traj,
        h_y_traj,
        h_vel_traj,
    ):
        """
        Parameters
        ----------
        h_x_traj   : (num_humans, num)  predicted x positions of each human over horizon
        h_y_traj   : (num_humans, num)  predicted y positions
        h_vel_traj : (num_humans, num, 2)  predicted [vx, vy] of each human over horizon
        """
        cov_v_control = 60 * jnp.identity(self.nvar)
        cov_psidot_control = 40 * jnp.identity(self.nvar)
        cov_control = jax.scipy.linalg.block_diag(cov_v_control, cov_psidot_control)

        c_v_samples, c_psidot_samples, key = self.compute_control_samples(
            key, mean_control, cov_control
        )

        # Simple MPPI constraints
        c_v_samples = jnp.clip(c_v_samples, self.v_min, self.v_max)
        c_psidot_samples = jnp.clip(
            c_psidot_samples, self.psi_dot_min, self.psi_dot_max
        )

        v_samples = jnp.dot(self.P_jax, c_v_samples.T).T
        psidot_samples = jnp.dot(self.P_jax, c_psidot_samples.T).T

        x_roll, y_roll, psi_roll = self.compute_rollouts(
            x_init, y_init, psi_init, v_samples, psidot_samples
        )

        primitives = jnp.stack([x_roll, y_roll], axis=-1)
        cost_tracking = self.compute_perpendicular_errors(global_path, primitives)

        controls_stack = jnp.stack((v_samples, psidot_samples), axis=-1)

        # Per-sample total cost
        cost = self.compute_cost_mppi_batch(
            controls_stack,
            x_roll,
            y_roll,
            x_fin,
            y_fin,
            x_obs,
            y_obs,
            h_x_traj,
            h_y_traj,
            h_vel_traj,
        )

        goal_cost2 = (
            (x_fin - x_roll[:, -1]) ** 2 + (y_fin - y_roll[:, -1]) ** 2
        ) * self.w_4
        goal_cost = self.goal_reaching_batch(x_roll, y_roll, x_fin, y_fin)

        S = cost + self.w_1 * goal_cost + goal_cost2 + self.w_5 * cost_tracking

        rho = S.min()
        eta = jnp.sum(jnp.exp((-1.0 / self.param_lambda) * (S - rho)))
        w = self.compute_weights_batch(S, rho, eta)

        epsilon_stack = controls_stack - jnp.mean(controls_stack, axis=0)
        w_epsilon = self.compute_epsilon_batch(epsilon_stack, w)
        samples_new = jnp.mean(controls_stack, axis=0) + w_epsilon

        v_samples_new = samples_new[:, 0]
        samples_psidot_new = samples_new[:, 1]

        c_v_new = (
            jnp.linalg.inv(self.P_jax.T @ self.P_jax + 0.0001 * jnp.identity(self.num))
            @ self.P_jax.T
            @ v_samples_new
        )
        c_psidot_new = (
            jnp.linalg.inv(self.P_jax.T @ self.P_jax + 0.0001 * jnp.identity(self.num))
            @ self.P_jax.T
            @ samples_psidot_new
        )

        # Single projection for optimal control
        # Bounded simple MPPI
        c_v_single = jnp.clip(c_v_new, self.v_min, self.v_max)
        c_psidot_single = jnp.clip(c_psidot_new, self.psi_dot_min, self.psi_dot_max)

        v_single = jnp.dot(self.P_jax, c_v_single.T).T
        psidot_single = jnp.dot(self.P_jax, c_psidot_single.T).T

        x_roll_opt, y_roll_opt, psi_roll_opt = self.compute_rollouts_single(
            x_init, y_init, psi_init, v_single, psidot_single
        )

        mean_new = jnp.hstack((c_v_single, c_psidot_single))

        v_next = jnp.mean(v_single[0:3])
        psidot_next = jnp.mean(psidot_single[0:3])
        psi_next = psi_init + psidot_next * self.dt
        vdot_next = (v_next - v_init) / self.dt
        psiddot_next = (psidot_next - psi_dot_init) / self.dt

        v_dot, v_ddot, psi_dot_arr, psi_ddot = self.dot_controls(
            v_single, psidot_single, v_init, psi_init, v_dot_init, psi_dot_init
        )

        v_ddot_next = v_ddot[3]
        psi_dddot_next = psi_ddot[3]

        return (
            mean_new,
            x_roll_opt,
            y_roll_opt,
            key,
            v_next,
            psi_next,
            psidot_next,
            vdot_next,
            psiddot_next,
            v_ddot_next,
            psi_dddot_next,
            v_single,
            psidot_single,
            x_roll,
            y_roll,
            w,
        )  # expose sample trajs + weights for viz


# Human (Pedestrian) dynamics


class Human:
    """
    Constant-velocity pedestrian that orbits or crosses the scene.
    Optionally follows a circular orbit (gives nice cross-traffic).
    """

    def __init__(self, x0, y0, vx, vy, mode="linear", cx=0.0, cy=0.0, r=0.0, omega=0.0):
        self.x = x0
        self.y = y0
        self.vx = vx
        self.vy = vy
        self.mode = mode  # 'linear' | 'circular'
        self.cx = cx
        self.cy = cy
        self.r = r
        self.omega = omega  # angular velocity [rad/s]
        self.theta = np.arctan2(y0 - cy, x0 - cx)
        self.trail_x = [x0]
        self.trail_y = [y0]

    def step(self, dt):
        if self.mode == "circular":
            self.theta += self.omega * dt
            self.x = self.cx + self.r * np.cos(self.theta)
            self.y = self.cy + self.r * np.sin(self.theta)
            self.vx = -self.r * self.omega * np.sin(self.theta)
            self.vy = self.r * self.omega * np.cos(self.theta)
        else:
            self.x += self.vx * dt
            self.y += self.vy * dt
        self.trail_x.append(self.x)
        self.trail_y.append(self.y)

    def predict(self, dt, horizon):
        """Linear-constant-velocity prediction over horizon steps."""
        xs, ys, vxs, vys = [], [], [], []
        px, py = self.x, self.y
        pvx, pvy = self.vx, self.vy
        if self.mode == "circular":
            theta = self.theta
            for _ in range(horizon):
                theta += self.omega * dt
                px = self.cx + self.r * np.cos(theta)
                py = self.cy + self.r * np.sin(theta)
                pvx = -self.r * self.omega * np.sin(theta)
                pvy = self.r * self.omega * np.cos(theta)
                xs.append(px)
                ys.append(py)
                vxs.append(pvx)
                vys.append(pvy)
        else:
            for _ in range(horizon):
                px += pvx * dt
                py += pvy * dt
                xs.append(px)
                ys.append(py)
                vxs.append(pvx)
                vys.append(pvy)
        return np.array(xs), np.array(ys), np.array(vxs), np.array(vys)


# Gaussian field visualisation helper (numpy, for plotting)


def gaussian_field_np(X, Y, hx, hy, hvx, hvy):
    speed = np.sqrt(hvx**2 + hvy**2) + 1e-6
    ux, uy = hvx / speed, hvy / speed
    nx, ny = -uy, ux
    dx = X - hx
    dy = Y - hy
    d_along = dx * ux + dy * uy
    d_perp = dx * nx + dy * ny
    sigma_perp = 0.6
    sigma_fwd = 1.2 + 0.5 * speed
    sigma_bwd = 0.5
    sigma_along = np.where(d_along >= 0, sigma_fwd, sigma_bwd)
    exponent = 0.5 * ((d_along / sigma_along) ** 2 + (d_perp / sigma_perp) ** 2)
    return np.exp(-exponent)


# Scene setup


def build_scene(randomize=False, difficulty="hard"):
    """
    Corridor scene: ego drives along x-axis.  Humans cross or orbit around it.
    """
    if randomize:
        humans = []

        if difficulty == "easy":
            num_linear, num_circular = 1, 0
        elif difficulty == "medium":
            num_linear, num_circular = 3, 1
        else:  # hard
            num_linear, num_circular = 5, 2

        # Cross-traffic pedestrians
        for _ in range(num_linear):
            x0 = np.random.uniform(1.0, 12.0)
            if np.random.rand() > 0.5:
                y0 = np.random.uniform(2.5, 4.0)
                vy = np.random.uniform(-1.2, -0.5)
            else:
                y0 = np.random.uniform(-4.0, -2.5)
                vy = np.random.uniform(0.5, 1.2)
            vx = np.random.uniform(-0.1, 0.1)
            humans.append(Human(x0=x0, y0=y0, vx=vx, vy=vy, mode="linear"))

        # Orbiting pedestrians
        for _ in range(num_circular):
            cx = np.random.uniform(4.0, 10.0)
            cy = np.random.uniform(-1.0, 1.0)
            r = np.random.uniform(1.2, 2.0)
            omega = np.random.uniform(0.4, 0.8) * np.random.choice([-1, 1])
            theta0 = np.random.uniform(0, 2 * np.pi)
            x0 = cx + r * np.cos(theta0)
            y0 = cy + r * np.sin(theta0)
            humans.append(
                Human(
                    x0=x0,
                    y0=y0,
                    vx=0.0,
                    vy=0.0,
                    mode="circular",
                    cx=cx,
                    cy=cy,
                    r=r,
                    omega=omega,
                )
            )
    else:
        humans = [
            # Cross-traffic: 4 pedestrians walking across the corridor
            Human(x0=-1.0, y0=3.5, vx=0.0, vy=-0.9, mode="linear"),
            Human(x0=3.0, y0=-3.0, vx=0.0, vy=0.8, mode="linear"),
            Human(x0=7.0, y0=3.0, vx=0.0, vy=-0.7, mode="linear"),
            Human(x0=11.0, y0=-3.5, vx=0.0, vy=1.0, mode="linear"),
            # Two pedestrians orbiting around midpoint of corridor
            Human(
                x0=5.0,
                y0=1.5,
                vx=0.0,
                vy=0.0,
                mode="circular",
                cx=5.0,
                cy=0.0,
                r=1.8,
                omega=0.5,
            ),
            Human(
                x0=5.0,
                y0=-1.5,
                vx=0.0,
                vy=0.0,
                mode="circular",
                cx=5.0,
                cy=0.0,
                r=1.8,
                omega=-0.5,
            ),
            # A slow stroller moving ahead of the ego
            Human(x0=2.0, y0=0.3, vx=0.4, vy=0.05, mode="linear"),
        ]

    # Global reference path: straight corridor along y≈0
    global_path = np.array([[i, 0.0] for i in np.linspace(0, 15, 30)])

    # Static obstacles (none in this scene, but API expects arrays)
    x_obs_static = jnp.array([])
    y_obs_static = jnp.array([])

    return humans, global_path, x_obs_static, y_obs_static


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────


def run_simulation(
    seed=0,
    headless=False,
    max_steps=200,
    randomize=False,
    use_gaussian_cost=True,
    difficulty="hard",
):
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed + 42)

    num_batch = 512
    planner = proj_mppi(num_batch)
    if not use_gaussian_cost:
        planner.w_6 = 0.0

    humans, global_path, x_obs_static, y_obs_static = build_scene(
        randomize=randomize, difficulty=difficulty
    )
    num_humans = len(humans)

    # If no static obstacles, provide dummy arrays with shape (1,) so vmap works
    if x_obs_static.shape[0] == 0:
        x_obs_static = jnp.array([999.0])
        y_obs_static = jnp.array([999.0])

    global_path_jax = jnp.array(global_path)

    # Ego initial state
    x_ego, y_ego = 0.0, 0.0
    psi_ego = 0.0
    v_ego = 0.5
    vdot_ego = 0.0
    psidot_ego = 0.0
    psiddot_ego = 0.0
    psi_dddot_ego = 0.0

    x_goal, y_goal = 14.0, 0.0

    mean_control = jnp.zeros(2 * planner.nvar)

    sim_dt = planner.dt
    max_steps = 200
    goal_radius = 0.5

    # Storage for animation
    ego_traj_x, ego_traj_y = [x_ego], [y_ego]
    frames = []  # list of dicts for animation

    print("Warming up JAX JIT...")
    if not headless:
        fig, update_plot = setup_realtime_plot(global_path, humans, x_goal, y_goal)
    # Build dummy human arrays for warm-up
    h_x_dummy = jnp.zeros((num_humans, planner.num))
    h_y_dummy = jnp.zeros((num_humans, planner.num))
    h_vel_dummy = jnp.zeros((num_humans, planner.num, 2))

    if not use_gaussian_cost and len(humans) > 0:
        x_obs_warmup = jnp.array([h.x for h in humans])
        y_obs_warmup = jnp.array([h.y for h in humans])
    else:
        x_obs_warmup = x_obs_static
        y_obs_warmup = y_obs_static

    _ = planner.compute_mppi(
        v_ego,
        vdot_ego,
        psi_ego,
        psidot_ego,
        psiddot_ego,
        x_ego,
        y_ego,
        x_goal,
        y_goal,
        mean_control,
        key,
        x_obs_warmup,
        y_obs_warmup,
        global_path_jax,
        h_x_dummy,
        h_y_dummy,
        h_vel_dummy,
    )
    print("Warm-up done. Running simulation...")

    t0 = time.time()
    for step in range(max_steps):
        # ── Predict human positions over planning horizon ──────────────────
        pred_hx = np.zeros((num_humans, planner.num))
        pred_hy = np.zeros((num_humans, planner.num))
        pred_hvx = np.zeros((num_humans, planner.num))
        pred_hvy = np.zeros((num_humans, planner.num))

        for hi, h in enumerate(humans):
            xs, ys, vxs, vys = h.predict(sim_dt, planner.num)
            pred_hx[hi] = xs
            pred_hy[hi] = ys
            pred_hvx[hi] = vxs
            pred_hvy[hi] = vys

        h_x_jax = jnp.array(pred_hx)
        h_y_jax = jnp.array(pred_hy)
        h_vel_jax = jnp.stack([jnp.array(pred_hvx), jnp.array(pred_hvy)], axis=-1)

        # ── MPPI solve ─────────────────────────────────────────────────────
        if not use_gaussian_cost and len(humans) > 0:
            x_obs_run = jnp.array([h.x for h in humans])
            y_obs_run = jnp.array([h.y for h in humans])
        else:
            x_obs_run = x_obs_static
            y_obs_run = y_obs_static

        result = planner.compute_mppi(
            v_ego,
            vdot_ego,
            psi_ego,
            psidot_ego,
            psiddot_ego,
            x_ego,
            y_ego,
            x_goal,
            y_goal,
            mean_control,
            key,
            x_obs_run,
            y_obs_run,
            global_path_jax,
            h_x_jax,
            h_y_jax,
            h_vel_jax,
        )

        (
            mean_control,
            x_opt,
            y_opt,
            key,
            v_next,
            psi_next,
            psidot_next,
            vdot_next,
            psiddot_next,
            v_ddot_next,
            psi_dddot_next,
            v_single,
            psidot_single,
            x_samp,
            y_samp,
            weights,
        ) = result

        # ── Advance ego ────────────────────────────────────────────────────
        x_ego = float(x_opt[1])
        y_ego = float(y_opt[1])
        psi_ego = float(psi_next)
        v_ego = float(v_next)
        vdot_ego = float(vdot_next)
        psidot_ego = float(psidot_next)
        psiddot_ego = float(psiddot_next)

        ego_traj_x.append(x_ego)
        ego_traj_y.append(y_ego)

        # ── Advance humans ─────────────────────────────────────────────────
        for h in humans:
            h.step(sim_dt)

        # ── Snapshot for animation ─────────────────────────────────────────
        frames.append(
            {
                "ego_x": x_ego,
                "ego_y": y_ego,
                "psi_ego": float(psi_ego),
                "opt_x": np.array(x_opt),
                "opt_y": np.array(y_opt),
                "samp_x": np.array(x_samp),
                "samp_y": np.array(y_samp),
                "weights": np.array(weights),
                "humans": [(h.x, h.y, h.vx, h.vy) for h in humans],
                "ego_trail_x": list(ego_traj_x),
                "ego_trail_y": list(ego_traj_y),
            }
        )

        dist_to_goal = np.hypot(x_ego - x_goal, y_ego - y_goal)
        if not headless:
            update_plot(frames[-1], step)

        # ── Collision Check ────────────────────────────────────────────────
        collision = False
        for h in humans:
            if (
                np.hypot(x_ego - h.x, y_ego - h.y) <= 0.55
            ):  # Ego radius 0.25 + Human radius 0.30
                print(f"Collision with human at step {step}! Simulation ends.")
                collision = True
                break

        if collision:
            break

        if dist_to_goal < goal_radius:
            print(f"Goal reached at step {step}! Distance: {dist_to_goal:.3f}")
            break

    elapsed = time.time() - t0
    print(
        f"Simulation done: {len(frames)} steps in {elapsed:.2f}s "
        f"({len(frames) / elapsed:.1f} Hz)"
    )
    if not headless:
        plt.ioff()
        plt.show()

    return frames, global_path, humans, x_goal, y_goal, collision, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Animation
# ─────────────────────────────────────────────────────────────────────────────


def setup_realtime_plot(global_path, humans_ref, x_goal, y_goal):
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0d0d1a")
    ax_main, ax_field = axes

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#444466")

    # Colour palette
    C_EGO = "#00e5ff"
    C_PATH = "#2979ff"
    C_OPT = "#ffffff"
    C_TRAIL = "#00e5ff"
    C_SAMP = "#444488"
    C_HUMAN = "#ff6b35"
    C_GOAL = "#76ff03"

    #  Grid for Gaussian field
    Gx = np.linspace(-2, 16, 280)
    Gy = np.linspace(-5, 5, 280)
    GX, GY = np.meshgrid(Gx, Gy)

    # Static elements
    for ax in axes:
        ax.plot(
            global_path[:, 0],
            global_path[:, 1],
            "--",
            color=C_PATH,
            lw=1.5,
            alpha=0.5,
            label="Global path",
        )
        ax.plot(x_goal, y_goal, "*", color=C_GOAL, ms=14, zorder=10, label="Goal")
        ax.set_xlim(-2, 16)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.legend(
            loc="upper right",
            fontsize=7,
            facecolor="#0d0d1a",
            labelcolor="white",
            framealpha=0.6,
        )

    ax_main.set_title(
        "MPPI Planner — Sample Trajectories", color="white", fontsize=11, pad=8
    )
    ax_field.set_title(
        "Human 2D Gaussian Cost Field", color="white", fontsize=11, pad=8
    )

    # Dynamic artists — main panel
    samp_lines = [
        ax_main.plot([], [], "-", color=C_SAMP, lw=0.4, alpha=0.3)[0] for _ in range(80)
    ]
    (opt_line,) = ax_main.plot(
        [], [], "-", color=C_OPT, lw=2.5, zorder=8, label="Optimal traj"
    )
    (trail_line,) = ax_main.plot(
        [], [], "-", color=C_TRAIL, lw=1.5, alpha=0.7, zorder=7
    )
    ego_patch = plt.Circle((0, 0), 0.25, color=C_EGO, zorder=9)
    ax_main.add_patch(ego_patch)
    ego_arrow = FancyArrowPatch(
        (0, 0), (0.6, 0), color=C_EGO, arrowstyle="->", mutation_scale=10, zorder=10
    )
    ax_main.add_patch(ego_arrow)

    human_circles_main = [
        plt.Circle((0, 0), 0.3, color=C_HUMAN, alpha=0.85, zorder=8) for _ in humans_ref
    ]
    human_arrows_main = [
        FancyArrowPatch(
            (0, 0),
            (0.5, 0),
            color="#ffcc00",
            arrowstyle="->",
            mutation_scale=8,
            zorder=9,
        )
        for _ in humans_ref
    ]
    for c in human_circles_main:
        ax_main.add_patch(c)
    for a in human_arrows_main:
        ax_main.add_patch(a)

    # Dynamic artists — field panel
    field_img = ax_field.imshow(
        np.zeros_like(GX),
        origin="lower",
        extent=[Gx[0], Gx[-1], Gy[0], Gy[-1]],
        cmap="hot",
        vmin=0,
        vmax=1.0,
        alpha=0.75,
        aspect="auto",
    )
    (opt_line2,) = ax_field.plot(
        [], [], "-", color=C_OPT, lw=2.5, zorder=8, label="Optimal traj"
    )
    (trail_line2,) = ax_field.plot(
        [], [], "-", color=C_TRAIL, lw=1.5, alpha=0.7, zorder=7
    )
    ego_patch2 = plt.Circle((0, 0), 0.25, color=C_EGO, zorder=9)
    ax_field.add_patch(ego_patch2)
    human_circles_field = [
        plt.Circle((0, 0), 0.3, color=C_HUMAN, alpha=0.85, zorder=8) for _ in humans_ref
    ]
    for c in human_circles_field:
        ax_field.add_patch(c)

    step_text = ax_main.text(
        0.02, 0.96, "", transform=ax_main.transAxes, color="white", fontsize=9, va="top"
    )
    plt.tight_layout(pad=1.5)

    def update(fr, frame_idx):
        # ── Sample trajectories (colour by weight) ────────────────────────
        ws = fr["weights"]
        w_norm = (ws - ws.min()) / (ws.max() - ws.min() + 1e-8)
        n_show = len(samp_lines)
        idx_sort = np.argsort(w_norm)[:n_show]
        for k, si in enumerate(idx_sort):
            samp_lines[k].set_data(fr["samp_x"][si], fr["samp_y"][si])
            alpha = 0.1 + 0.5 * float(w_norm[si])
            samp_lines[k].set_alpha(alpha)

        # ── Optimal trajectory ────────────────────────────────────────────
        opt_line.set_data(fr["opt_x"], fr["opt_y"])
        opt_line2.set_data(fr["opt_x"], fr["opt_y"])

        # ── Ego ──────────────────────────────────────────────────────────
        ego_patch.center = (fr["ego_x"], fr["ego_y"])
        ego_patch2.center = (fr["ego_x"], fr["ego_y"])
        psi = fr["psi_ego"]
        dx_arrow = 0.55 * np.cos(psi)
        dy_arrow = 0.55 * np.sin(psi)
        ego_arrow.set_positions(
            (fr["ego_x"], fr["ego_y"]), (fr["ego_x"] + dx_arrow, fr["ego_y"] + dy_arrow)
        )

        # ── Trail ─────────────────────────────────────────────────────────
        trail_line.set_data(fr["ego_trail_x"], fr["ego_trail_y"])
        trail_line2.set_data(fr["ego_trail_x"], fr["ego_trail_y"])

        # ── Humans ────────────────────────────────────────────────────────
        total_field = np.zeros_like(GX)
        for hi, (hx, hy, hvx, hvy) in enumerate(fr["humans"]):
            human_circles_main[hi].center = (hx, hy)
            human_circles_field[hi].center = (hx, hy)
            spd = np.hypot(hvx, hvy) + 1e-6
            human_arrows_main[hi].set_positions(
                (hx, hy), (hx + 0.5 * hvx / spd, hy + 0.5 * hvy / spd)
            )
            total_field += gaussian_field_np(GX, GY, hx, hy, hvx, hvy)

        field_img.set_data(np.clip(total_field, 0, 1))

        step_text.set_text(f"Step {frame_idx:03d}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    return fig, update


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps for the simulation",
    )
    args = parser.parse_args()

    run_simulation(max_steps=args.max_steps)
