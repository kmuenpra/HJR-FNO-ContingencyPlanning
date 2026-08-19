"""
Environment for rrt_2D
@author: huiming zhou
"""
import numpy as np

class Env:
    def __init__(self, safe_regions=[], unknown_obs=None, x_range=None, y_range=None):
        """
        unknown_obs / x_range / y_range: optional overrides so a shared evaluation
            scenario (eval/scenarios.py) can define the world instead of the
            hard-coded defaults below. All default to None -> original behaviour.
        """
        self.x_range = (-25, 25) if x_range is None else tuple(x_range)
        self.y_range = (-25, 25) if y_range is None else tuple(y_range)
        self.safe_regions = safe_regions
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.unknown_obs_circle = (
            self.unknown_obs_circle()
            if unknown_obs is None
            else [list(o) for o in unknown_obs]
        )

        filtered_obs = []

        for ox, oy, orad in self.unknown_obs_circle:
            intersects = False

            for sx, sy, srad in self.safe_regions:
                d = np.hypot(ox - sx, oy - sy)

                if d < orad + srad:
                    intersects = True
                    break

            if not intersects:
                filtered_obs.append([ox, oy, orad])

        # A scenario-supplied list must survive this filter untouched, or RRTX and
        # MPPI are not solving the same problem: MPPI's env does NOT filter, so
        # anything dropped here exists for one planner and not the other. Random
        # layouts satisfy R2 by construction (eval/scenarios.py) so they never
        # trip it; the hand-authored ENV_C list is stored post-filter for the same
        # reason. Warn rather than raise, so the built-in default map (which is
        # meant to be filtered) still runs.
        if unknown_obs is not None and len(filtered_obs) != len(self.unknown_obs_circle):
            print(f"[env] WARNING: dropped "
                  f"{len(self.unknown_obs_circle) - len(filtered_obs)} of "
                  f"{len(self.unknown_obs_circle)} scenario obstacles for "
                  f"overlapping a safe region -- MPPI does not filter, so the two "
                  f"planners now see different maps.")

        self.unknown_obs_circle = filtered_obs

        
            

    @staticmethod
    def obs_boundary():
        obs_boundary = []
        # obs_boundary = [
        #     [-20, -20, 1, 40],   # left wall
        #     [-20, 20, 40, 1],    # top wall
        #     [20, -20, 1, 40],    # right wall
        #     [-20, -20, 40, 1]    # bottom wall
        # ]

        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            # [14, 12, 8, 2],
            # [18, 22, 8, 3],
            # [26, 7, 2, 12],
            # [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            # [7, 12, 3],
            # [46, 20, 2],
            # [15, 5, 2],
            # [37, 7, 3],
            # [37, 23, 3]
        ]

        return obs_cir
    
    @staticmethod
    def unknown_obs_circle():
        obs_cir = [
        [-5.0,   4.0,  1.5],
        [-6.0,  -6.0,  2.0],
        [ 7.0,   7.0,  1.0],
        [-10.0, -10.0, 1.8],
        [10, 6, 1.3],
        [-3.5, 7, 1.3],
        [ 0.0,  12.0,  1.6],
        [ -2.0, -14.0, 1.6],
        [  4.0,  -6.0, 1.4],
        [ 10.0,  -14.0, 1.3],
        [ 11.0,   -3.0, 1.4],
        [  6.0,  16.0, 1.6],
        [ -12.0,   10.0, 1.7],
        [ -14.0,  3.0, 1.7],
        [ -16.0, -4.0, 1.5], #[-16, -4]
        [ -9.0,  18.0, 1.8],
        [  3.0,   1.0, 1.2],
        [-2.5, -3.5, 1.3]
        ]
        return obs_cir


