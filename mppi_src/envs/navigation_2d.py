"""
Kohei Honda, 2023.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from envs.obstacle_map_2d import ObstacleMap, generate_random_obstacles


@torch.jit.script
def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi


class Navigation2DEnv:
    def __init__(
        self, device=torch.device("cuda"), dtype=torch.float32, seed: int = 42
    ) -> None:
        # device and dtype
        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        self._obstacle_map = ObstacleMap(
            map_size=(20, 20), cell_size=0.1, device=self._device, dtype=self._dtype
        )
        self._seed = seed

        # safe regions (visualization only, no effect on the planner): fixed
        # centers, defined before obstacles so obstacles can avoid them.
        safe_radius = 2.0
        safe_region_centers = [
            (-5.0, -7.5),
            (0.0, 0.0),
            (-2.5, 5.0),
            (5.0, 6.0),
        ]
        self.safe_regions = [(cx, cy, safe_radius) for cx, cy in safe_region_centers]

        # obstacles are constrained to not intersect any safe region
        generate_random_obstacles(
            obstacle_map=self._obstacle_map,
            random_x_range=(-7.5, 7.5),
            random_y_range=(-7.5, 7.5),
            num_circle_obs=10,
            radius_range=(0.5, 1.5),
            num_rectangle_obs=0,
            width_range=(2, 2),
            height_range=(2, 2),
            max_iteration=1000,
            seed=seed,
            keepout_circles=self.safe_regions,
        )

        # Treat all generated obstacles as UNKNOWN ground truth: snapshot them,
        # then clear the map so it starts with no KNOWN obstacles. The lidar
        # reveals obstacles incrementally (see step()), adding them to the map.
        self._unknown_obs = [
            (float(o.center[0]), float(o.center[1]), float(o.radius))
            for o in self._obstacle_map.circle_obs_list
        ]
        self._known_obs = []
        self._obstacle_map.clear()
        self._obstacle_map.convert_to_torch()

        # Expose the obstacle lists under the names the HJR-FNO Utils helper
        # (utils.py) reads in its constructor: Utils.__init__ copies these
        # references, and Utils.lidar_detected() mutates unknown_obs_circle in
        # place. They are the SAME list objects as _known_obs/_unknown_obs, so
        # the env, the Utils helper, and the renderer share one source of truth
        # for "what's been discovered so far". Only circular obstacles are
        # modeled here, so obs_rectangle/obs_boundary stay empty.
        self.obs_circle = self._known_obs
        self.unknown_obs_circle = self._unknown_obs
        self.obs_rectangle = []
        self.obs_boundary = []

        self._start_pos = torch.tensor(
            [-9.0, -9.0], device=self._device, dtype=self._dtype
        )
        self._goal_pos = torch.tensor(
            [8.0, 8.0], device=self._device, dtype=self._dtype
        )

        self._robot_state = torch.zeros(3, device=self._device, dtype=self._dtype)
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        # u: [v, omega] (m/s, rad/s)
        self.u_min = torch.tensor([0.0, -1.0], device=self._device, dtype=self._dtype)
        self.u_max = torch.tensor([1.0, 1.0], device=self._device, dtype=self._dtype)

        # lidar sensor: circular field of view centered on the robot
        self.lidar_radius = 7.0

        # reachability oracle (HJR-FNO), attached after construction via
        # attach_reachability(); step() feeds it newly detected obstacles.
        self.hjr_fno = None

    def attach_reachability(self, hjr_fno) -> None:
        """Attach the HJR-FNO reachability oracle so step() can update its
        reachable sets when the lidar detects new obstacles, and so detection
        itself runs through the oracle's Utils helper (utils.py).

        Utils.__init__ already copied this env's obstacle lists (so utils.
        obs_circle / unknown_obs_circle are the SAME list objects as ours); we
        (re)bind them explicitly and override the sensing radius so Utils.
        lidar_detected() uses the env's lidar range instead of its 2.0 m default.
        """
        self.hjr_fno = hjr_fno
        utils = getattr(hjr_fno, "utils", None)
        if utils is not None:
            utils.unknown_obs_circle = self.unknown_obs_circle
            utils.obs_circle = self.obs_circle
            utils.sensing_radius = self.lidar_radius

    def lidar_detected(self) -> list:
        """Unknown obstacles whose center lies within lidar_radius of the robot.

        Returns:
            list of (x, y, radius) tuples in world coordinates.
        """
        rx = self._robot_state[0].item()
        ry = self._robot_state[1].item()
        detected = []
        for ox, oy, r in self._unknown_obs:
            if np.hypot(ox - rx, oy - ry) <= self.lidar_radius:
                detected.append((ox, oy, r))
        return detected

    def reset(self) -> torch.Tensor:
        """
        Reset robot state.
        Returns:
            torch.Tensor: shape (3,) [x, y, theta]
        """
        self._robot_state[:2] = self._start_pos
        self._robot_state[2] = angle_normalize(
            torch.atan2(
                self._goal_pos[1] - self._start_pos[1],
                self._goal_pos[0] - self._start_pos[0],
            )
        )

        self._fig = plt.figure(layout="tight")
        self._ax = self._fig.add_subplot()
        self._ax.set_xlim(self._obstacle_map.x_lim)
        self._ax.set_ylim(self._obstacle_map.y_lim)
        self._ax.set_aspect("equal")

        self._rendered_frames = []

        return self._robot_state

    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Update robot state based on differential drive dynamics.
        Args:
            u (torch.Tensor): control batch tensor, shape (2) [v, omega]
        Returns:
            Tuple[torch.Tensor, bool]: Tuple of robot state and is goal reached.
        """
        u = torch.clamp(u, self.u_min, self.u_max)

        self._robot_state = self.dynamics(
            state=self._robot_state.unsqueeze(0), action=u.unsqueeze(0)
        ).squeeze(0)

        # lidar detection: reveal unknown obstacles within range. When the
        # oracle is attached, detection is delegated to its Utils helper
        # (utils.py) so the env, the oracle, and the renderer share one source
        # of truth; Utils.lidar_detected() removes the detected obstacles from
        # unknown_obs_circle in place (the same list object as _unknown_obs).
        # Without an oracle we fall back to the env's own range check.
        robot_xy = (self._robot_state[0].item(), self._robot_state[1].item())
        utils = getattr(self.hjr_fno, "utils", None) if self.hjr_fno is not None else None
        if utils is not None:
            _, detected = utils.lidar_detected(robot_xy)
        else:
            detected = self.lidar_detected()
            if detected:
                detected_set = set(detected)
                # mutate the unknown list in place to keep its shared identity
                self._unknown_obs[:] = [
                    o for o in self._unknown_obs if o not in detected_set
                ]

        if detected:
            # newly seen obstacles move unknown -> known: rasterize into the
            # obstacle map (so the MPPI cost sees them) and report to the oracle.
            for ox, oy, r in detected:
                self._known_obs.append((ox, oy, r))
                self._obstacle_map.add_circle_obstacle(np.array([ox, oy]), r)
            self._obstacle_map.convert_to_torch()
            if self.hjr_fno is not None:
                self.hjr_fno.update_obs(detected)

        # goal check
        goal_threshold = 0.5
        is_goal_reached = (
            torch.norm(self._robot_state[:2] - self._goal_pos) < goal_threshold
        )

        return self._robot_state, is_goal_reached

    def render(
        self,
        predicted_trajectory: torch.Tensor = None,
        is_collisions: torch.Tensor = None,
        top_samples: Tuple[torch.Tensor, torch.Tensor] = None,
        mode: str = "human",
        overlay_fn=None,
    ) -> None:
        self._ax.set_xlabel("x [m]")
        self._ax.set_ylabel("y [m]")

        # safe regions
        for cx, cy, radius in self.safe_regions:
            self._ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    radius,
                    color="green",
                    alpha=0.3,
                    zorder=5,
                )
            )

        # unknown (not-yet-detected) obstacles: red dashed circles
        for ox, oy, r in self._unknown_obs:
            self._ax.add_patch(
                plt.Circle(
                    (ox, oy),
                    r,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    linewidth=1.5,
                    zorder=9,
                )
            )

        # obstacle map (known/detected obstacles)
        self._obstacle_map.render(self._ax, zorder=10)

        # start and goal
        self._ax.scatter(
            self._start_pos[0].item(),
            self._start_pos[1].item(),
            marker="o",
            color="red",
            zorder=10,
        )
        self._ax.scatter(
            self._goal_pos[0].item(),
            self._goal_pos[1].item(),
            marker="o",
            color="orange",
            zorder=10,
        )

        # robot
        self._ax.scatter(
            self._robot_state[0].item(),
            self._robot_state[1].item(),
            marker="o",
            color="green",
            zorder=100,
        )

        # lidar sensing range (dashed magenta circle around the robot)
        self._ax.add_patch(
            plt.Circle(
                (self._robot_state[0].item(), self._robot_state[1].item()),
                self.lidar_radius,
                edgecolor="cyan",
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
                zorder=50,
            )
        )

        # visualize top samples with different alpha based on weights
        if top_samples is not None:
            top_samples, top_weights = top_samples
            top_samples = top_samples.cpu().numpy()
            top_weights = top_weights.cpu().numpy()
            top_weights = 0.7 * top_weights / np.max(top_weights)
            top_weights = np.clip(top_weights, 0.1, 0.7)
            for i in range(top_samples.shape[0]):
                self._ax.plot(
                    top_samples[i, :, 0],
                    top_samples[i, :, 1],
                    color="magenta",
                    alpha=top_weights[i],
                    zorder=1,
                )

        # predicted trajectory
        if predicted_trajectory is not None:
            # if is collision color is red
            colors = np.array(["darkblue"] * predicted_trajectory.shape[1])
            if is_collisions is not None:
                is_collisions = is_collisions.cpu().numpy()
                is_collisions = np.any(is_collisions, axis=0)
                colors[is_collisions] = "red"

            self._ax.scatter(
                predicted_trajectory[0, :, 0].cpu().numpy(),
                predicted_trajectory[0, :, 1].cpu().numpy(),
                color=colors,
                marker="o",
                s=3,
                zorder=2,
            )

        # optional extra drawing on the axis (e.g. reachable-set overlays);
        # called here so it lands in the same frame, before pause/cla.
        if overlay_fn is not None:
            overlay_fn(self._ax)

        if mode == "human":
            # online rendering
            plt.pause(0.001)
            plt.cla()
        elif mode == "rgb_array":
            # offline rendering for video
            # TODO: high resolution rendering
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            # copy: buffer_rgba() is a view into the live canvas buffer, which
            # is overwritten on the next draw() (unlike the old tostring_rgb()
            # which returned a fresh bytes copy each call)
            data = data[..., :3].copy()  # drop alpha channel
            plt.cla()
            self._rendered_frames.append(data)

    def close(self, path: str = None) -> None:
        if path is None:
            # mkdir video if not exists

            if not os.path.exists("video"):
                os.mkdir("video")
            path = "video/" + "navigation_2d_" + str(self._seed) + ".gif"

        if len(self._rendered_frames) > 0:
            # save animation
            clip = ImageSequenceClip(self._rendered_frames, fps=10)
            # clip.write_videofile(path, fps=10)
            clip.write_gif(path, fps=10)

    def dynamics(
        self, state: torch.Tensor, action: torch.Tensor, delta_t: float = 0.1
    ) -> torch.Tensor:
        """
        Update robot state based on differential drive dynamics.
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
            delta_t (float): time step interval [s]
        Returns:
            torch.Tensor: shape (batch_size, 3) [x, y, theta]
        """

        # Perform calculations as before
        x = state[:, 0].view(-1, 1)
        y = state[:, 1].view(-1, 1)
        theta = state[:, 2].view(-1, 1)
        v = torch.clamp(action[:, 0].view(-1, 1), self.u_min[0], self.u_max[0])
        omega = torch.clamp(action[:, 1].view(-1, 1), self.u_min[1], self.u_max[1])
        theta = angle_normalize(theta)

        new_x = x + v * torch.cos(theta) * delta_t
        new_y = y + v * torch.sin(theta) * delta_t
        new_theta = angle_normalize(theta + omega * delta_t)

        # Clamp x and y to the map boundary
        x_lim = torch.tensor(
            self._obstacle_map.x_lim, device=self._device, dtype=self._dtype
        )
        y_lim = torch.tensor(
            self._obstacle_map.y_lim, device=self._device, dtype=self._dtype
        )
        clamped_x = torch.clamp(new_x, x_lim[0], x_lim[1])
        clamped_y = torch.clamp(new_y, y_lim[0], y_lim[1])

        result = torch.cat([clamped_x, clamped_y, new_theta], dim=1)

        return result

    def cost_function(
        self, state: torch.Tensor, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        """
        Calculate cost function
        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, 3) [x, y, theta]
            action (torch.Tensor): control batch tensor, shape (batch_size, 2) [v, omega]
        Returns:
            torch.Tensor: shape (batch_size,)
        """

        goal_cost = torch.norm(state[:, :2] - self._goal_pos, dim=1)

        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)

        obstacle_cost = self._obstacle_map.compute_cost(pos_batch).squeeze(
            1
        )  # (batch_size,)

        # HJR-FNO reachability constraint (optional, passed via info): a state is
        # infeasible if it lies outside its nearest safe region's reachable set.
        # Skipped when RBR is active: the reachability constraint is then enforced
        # by resampling (see MPPI._rollout_cost_rbr / points_safe), so re-checking
        # it here would only duplicate the expensive per-step reachability eval.
        hjr_fno = info.get("hjr_fno", None)
        if hjr_fno is not None and not info.get("rbr_active", False):
            points = state[:, :2].detach().cpu().numpy()  # (batch_size, 2) world xy
            thetas = state[:, 2].detach().cpu().numpy()  # (batch_size,) heading
            feasible = hjr_fno.points_feasible(points, thetas=thetas)  # (batch_size,) bool
            outside = torch.as_tensor(
                ~feasible, device=obstacle_cost.device
            )  # True where outside the reachable set
            obstacle_cost = torch.logical_or(obstacle_cost.bool(), outside).to(
                obstacle_cost.dtype
            )

        cost = goal_cost + 10000 * obstacle_cost

        return cost

    def points_safe(self, state: torch.Tensor, info: dict = None) -> torch.Tensor:
        """Hard constraint indicator for RBR: which states satisfy ALL state
        constraints (no collision AND inside the HJR-FNO reachable set).

        Args:
            state (torch.Tensor): state batch, shape (batch_size, 3) [x, y, theta]
            info (dict): unused (kept for the MPPI constraint_func signature).
        Returns:
            torch.BoolTensor: shape (batch_size,), True where the state is feasible.
        """
        pos_batch = state[:, :2].unsqueeze(1)  # (batch_size, 1, 2)
        collision = self._obstacle_map.compute_cost(pos_batch).squeeze(1) > 0.5
        safe = ~collision  # (batch_size,) bool on state's device

        if self.hjr_fno is not None:
            points = state[:, :2].detach().cpu().numpy()  # (batch_size, 2) world xy
            thetas = state[:, 2].detach().cpu().numpy()  # (batch_size,) heading
            feasible = self.hjr_fno.points_feasible(points, thetas=thetas)  # bool
            feasible = torch.as_tensor(feasible, device=safe.device, dtype=torch.bool)
            safe = safe & feasible

        return safe

    def collision_check(self, state: torch.Tensor) -> torch.Tensor:
        """

        Args:
            state (torch.Tensor): state batch tensor, shape (batch_size, traj_size , 3) [x, y, theta]
        Returns:
            torch.Tensor: shape (batch_size,)
        """
        pos_batch = state[:, :, :2]
        is_collisions = self._obstacle_map.compute_cost(pos_batch).squeeze(1)
        return is_collisions
