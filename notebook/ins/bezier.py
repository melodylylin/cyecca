import casadi as ca
from scipy.special import binom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from zipfile import ZipFile
import os
import datetime
# from .TimeOptBez import *
# from cyecca.lie.SE23 import *
from cyecca.models.bezier import derive_ref
import cyecca.lie as lie

f_ref = derive_ref()["f_ref"]

class Bezier:
#https://en.wikipedia.org/wiki/B%C3%A9zier_curve

    def __init__(self, P: ca.SX, T: float):
        self.P = P
        self.m = P.shape[0]
        self.n = P.shape[1]-1
        self.T = T
    
    def eval(self, t):
        #https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
        beta = t/self.T
        A = ca.SX(self.P)
        for j in range(1, self.n + 1):
            for k in range(self.n + 1 - j):
                A[:, k] = A[:, k] * (1 - beta) + A[:, k + 1] * beta
        return A[:, 0]
    
    def deriv(self, m=1):
        D = ca.SX(self.P)
        for j in range(0, m):
            D = (self.n - j)*ca.horzcat(*[ D[:, i+1] - D[:, i] for i in range(self.n - j) ])
        return Bezier(D/self.T**m, self.T)

def derive_bezier7():
    n = 8
    T = ca.SX.sym('T')
    t = ca.SX.sym('t')
    P = ca.SX.sym('P', 1, n)
    B = Bezier(P, T)

    # derivatives
    B_d = B.deriv()
    B_d2 = B_d.deriv()
    B_d3 = B_d2.deriv()
    B_d4 = B_d3.deriv()

    # boundary conditions

    # trajectory
    p = B.eval(t)
    v = B_d.eval(t)
    a = B_d2.eval(t)
    j = B_d3.eval(t)
    s = B_d4.eval(t)
    r = ca.vertcat(p, v, a, j, s)

    # given position/velocity boundary conditions, solve for bezier points
    wp_0 = ca.SX.sym('p0', 4, 1)  # pos/vel at waypoint 0
    wp_1 = ca.SX.sym('p1', 4, 1)  # pos/vel at waypoint 1

    constraints = []
    constraints += [(B.eval(0), wp_0[0])]  # pos @ wp0
    constraints += [(B_d.eval(0), wp_0[1])]  # vel @ wp0
    constraints += [(B_d2.eval(0), wp_0[2])]  # accel @ wp0
    constraints += [(B_d3.eval(0), wp_0[3])]  # jerk @ wp0
    constraints += [(B.eval(T), wp_1[0])]  # pos @ wp1
    constraints += [(B_d.eval(T), wp_1[1])]  # vel @ wp1
    constraints += [(B_d2.eval(T), wp_1[2])]  # accel @ wp1
    constraints += [(B_d3.eval(T), wp_1[3])]  # jerk @ wp1

    assert len(constraints) == n

    Y = ca.vertcat(*[c[0] for c in constraints])
    b = ca.vertcat(*[c[1] for c in constraints])
    A = ca.jacobian(Y, P)
    A_inv = ca.inv(A)
    P_sol = (A_inv@b).T

    functions = [
        ca.Function('bezier7_solve', [wp_0, wp_1, T], [P_sol],
            ['wp_0', 'wp_1', 'T'], ['P']),
        ca.Function('bezier7_traj', [t, T, P], [r],
            ['t', 'T', 'P'], ['r']),
    ]

    return { f.name(): f for f in functions }

def derive_bezier3():
    n = 4
    T = ca.SX.sym('T')
    t = ca.SX.sym('t')
    P = ca.SX.sym('P', 1, n)
    B = Bezier(P, T)

    # derivatives
    B_d = B.deriv()
    B_d2 = B_d.deriv()
    B_d3 = B_d2.deriv()
    B_d4 = B_d3.deriv()

    # boundary conditions

    # trajectory
    p = B.eval(t)
    v = B_d.eval(t)
    a = B_d2.eval(t)
    r = ca.vertcat(p, v, a)

    # given position/velocity boundary conditions, solve for bezier points
    wp_0 = ca.SX.sym('p0', 2, 1)  # pos/vel at waypoint 0
    wp_1 = ca.SX.sym('p1', 2, 1)  # pos/vel at waypoint 1

    constraints = []
    constraints += [(B.eval(0), wp_0[0])]  # pos @ wp0
    constraints += [(B_d.eval(0), wp_0[1])]  # vel @ wp0
    constraints += [(B.eval(T), wp_1[0])]  # pos @ wp1
    constraints += [(B_d.eval(T), wp_1[1])]  # vel @ wp1

    assert len(constraints) == n

    Y = ca.vertcat(*[c[0] for c in constraints])
    b = ca.vertcat(*[c[1] for c in constraints])
    A = ca.jacobian(Y, P)
    A_inv = ca.inv(A)
    P_sol = (A_inv@b).T

    functions = [
        ca.Function('bezier3_solve', [wp_0, wp_1, T], [P_sol],
            ['wp_0', 'wp_1', 'T'], ['P']),
        ca.Function('bezier3_traj', [t, T, P], [r],
            ['t', 'T', 'P'], ['r']),
    ]

    return { f.name(): f for f in functions }



# def multirotor_timeOpt(bc,k_time): ## Currently outputs optimized time
#     time_opt = find_opt_time(6, bc,k_time)
#     return np.average(time_opt)

def multirotor_plan(bc,T0, dt):
    bezier_7 = derive_bezier7()

    bc = np.array(bc)
    t0 = np.arange(0, T0, dt)

    PX = bezier_7['bezier7_solve'](bc[:, 0, 0], bc[:, 1, 0], T0)
    traj_x = np.array(bezier_7['bezier7_traj'](np.array([t0]), T0, PX)).T

    PY = bezier_7['bezier7_solve'](bc[:, 0, 1], bc[:, 1, 1], T0)
    traj_y = np.array(bezier_7['bezier7_traj'](np.array([t0]), T0, PY)).T

    PZ = bezier_7['bezier7_solve'](bc[:, 0, 2], bc[:, 1, 2], T0)
    traj_z = np.array(bezier_7['bezier7_traj'](np.array([t0]), T0, PZ)).T

    # V = np.sqrt(vx**2 + vy**2)bc_t = np.array([
    #         [ # position
    #         [0, 0, 0],  
    #         [0, 0, 2],    
    #         [0, 0, 5], 
    #         [120, -3.05, 5],
    #         [160, -5.2, 5],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 4],
    #         [229.7, -9.6, 2],
    #         ],
    #         [ # velocity
    #         [0, 0, 0.1],  
    #         [0, 0, 0.1], 
    #         [0, 0, 0],
    #         [2.2, -0.1, 0],
    #         [2.2, -0.1, 0],
    #         [2.2, -0.1, 0],
    #         [0, 0, 0],  # wp0, x, y, z
    #         [0, 0, 0],
    #         ],
    #         [ # accel
    #         [0, 0, 0], 
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ],
    #         [ # jerk
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #         ]])

    return PX, PY, PZ, traj_x, traj_y, traj_z, t0

def generate_path(bc_t, dt):
    
    t_total = 0
    res = {
        'anchor_x': [],
        'anchor_y': [],
        'anchor_z': [],
        'x': [],
        'y': [],
        'z': [],
        'v_x': [],
        'v_y': [],
        'v_z': [],
        'acc_x': [],
        'acc_y': [],
        'acc_z': [],
        'theta0': [],
        'theta1': [],
        'theta2': [],
        'T': [],
        'T0': [],
        'omega_1': [],
        'omega_2': [],
        'omega_3': []
    }

    for i in range(bc_t.shape[1]-1):
        bc = bc_t[:,i:i+2,:]
        T0 = 10
        Px, Py, Pz, traj_x, traj_y, traj_z, t0 = multirotor_plan(bc, T0, dt)
        t = t_total + t0
        t_total = t_total + T0 
        x = traj_x[:, 0]
        y = traj_y[:, 0]
        z = traj_z[:, 0]
        vx = traj_x[:, 1]
        vy = traj_y[:, 1]
        vz = traj_z[:, 1]
        ax = traj_x[:, 2]
        ay = traj_y[:, 2]
        az = traj_z[:, 2]-9.8
        jx = traj_x[:, 3]
        jy = traj_y[:, 3]
        jz = traj_z[:, 3]
        sx = traj_x[:, 4]
        sy = traj_y[:, 4]
        sz = traj_z[:, 4]

        romega1 = []
        romega2 = []
        romega3 = []
        rtheta0 = []
        rtheta1 = []
        rtheta2 = []
    
        for j in range(x.shape[0]):
            r_vx = vx[j]
            r_vy = vy[j]
            r_vz = vz[j]
            r_ax = ax[j]
            r_ay = ay[j]
            r_az = az[j]
            r_jx = jx[j]
            r_jy = jy[j]
            r_jz = jz[j]
            r_sx = sx[j]
            r_sy = sy[j]
            r_sz = sz[j]
            ref_v = f_ref(0, 0, 0, [r_vx, r_vy, r_vz], [r_ax, r_ay, r_az], [r_jx, r_jy, r_jz], [r_sx, r_sy, r_sz])
            quat = lie.SO3Quat.elem(ref_v[1])
            theta = ca.DM(lie.SO3Mrp.from_Quat(quat).param)
            theta = np.array(theta).reshape(3,)
            r_theta0 = theta[0]
            r_theta1 = theta[1]
            r_theta2 = theta[2]
            rtheta0.append(r_theta0)
            rtheta1.append(r_theta1)
            rtheta2.append(r_theta2)
            omega = ref_v[2]
            omega = np.array(omega).reshape(3,)
            r_omega1 = omega[0]
            r_omega2 = omega[1]
            r_omega3 = omega[2]
            romega1.append(r_omega1)
            romega2.append(r_omega2)
            romega3.append(r_omega3)

        res['anchor_x'].extend(np.array(Px).tolist())
        res['anchor_y'].extend(np.array(Py).tolist())
        res['anchor_z'].extend(np.array(Pz).tolist())
        res['x'].extend(x)
        res['y'].extend(y)
        res['z'].extend(z)
        res['theta0'].extend(np.array(rtheta0).tolist())
        res['theta1'].extend(np.array(rtheta1).tolist())
        res['theta2'].extend(np.array(rtheta2).tolist())
        res['v_x'].extend(vx)
        res['v_y'].extend(vy)
        res['v_z'].extend(vz)
        res['acc_x'].extend(ax)
        res['acc_y'].extend(ay)
        res['acc_z'].extend(az)
        res['T'].extend(np.array(t).tolist())
        res['T0'].append(T0)
        res['omega_1'].extend(np.array(romega1).tolist())
        res['omega_2'].extend(np.array(romega2).tolist())
        res['omega_3'].extend(np.array(romega3).tolist())

    return res