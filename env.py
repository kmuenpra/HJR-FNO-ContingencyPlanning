"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self, safe_regions=[]):
        self.x_range = (-25, 25)
        self.y_range = (-25, 25)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()
        self.unknown_obs_circle = self.unknown_obs_circle()
        self.safe_regions = safe_regions

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
        # [-6.0,  -6.0,  2.0],
        # [ 7.0,   7.0,  1.0],
        [-10.0, -10.0, 1.8],
        [ 0.0,  12.0,  2.0],
        [ -2.0, -14.0, 1.6],
        [  4.0,  -6.0, 1.4],
        [ 10.0,  -12.0, 1.8],
        [ 14.0,   2.0, 1.5],
        [  6.0,  16.0, 1.6],
        [ -11.0,   10.0, 1.5],
        [ -14.0,  3.0, 1.7],
        [ -16.0, -4.0, 1.5],
        [ -9.0,  18.0, 1.8],
        [  3.0,   1.0, 1.2],
        ]
        return obs_cir


