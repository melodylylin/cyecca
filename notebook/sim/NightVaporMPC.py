import casadi as ca
import numpy as np

def shift_timestep(h, state, control, f, p):
    delta_state = f(state, control[:, 0], p)
    next_state = ca.DM.full(state + h * delta_state)
    next_control = ca.horzcat(control[:, 1:],
                                  ca.reshape(control[:, -1], -1, 1))

    return next_state, next_control

def dm_to_array(dm):
    return np.array(dm.full())

def derive_dynamics():
    # p, parameters
    # parameters p: [tau_w, tau_r, c_t, m, J_z, rho, S]
    L = ca.SX.sym('L') # motor time constant
    p = ca.vertcat(L) # parameter vector
    
    
    p_defaults = {
        "L": 1, 
    }

    # x, state
    # states x: [w, r, vx, vy, omega, px, py, theta]
    p_x = ca.SX.sym('p_x') # x pos
    p_y = ca.SX.sym('p_y') # y pos
    theta = ca.SX.sym('theta') # vehicle heading angle about z
    x = ca.vertcat(p_x, p_y, theta) #state

    x0_defaults = {
        "p_x": 0,
        "p_y": 0,
        "theta": 0
    }

    # u, input
    # input u: [w_c, r_c]
    v = ca.SX.sym('v') # forward vel
    r_c = ca.SX.sym('r_c') # rudder deflection command
    u = ca.vertcat(v, r_c)

    # f = rhs d/dt x = f(x, u)
    x_dot = ca.vertcat(
        v*ca.cos(theta),
        v*ca.sin(theta),
        v/L*ca.tan(r_c)
    )
    
    f = ca.Function("f", [x, u, p], [x_dot], ["x", "u", "p"], ["x_dot"])
    return locals()

## single shooting method
# x0 is known
# know x1_p = f(x0, u0)
#
# find optimal u0 such that
# x1 = xt  (for simple case xt = x0)

## multiple shooting method
# x0 is known
# know x1_p = f(x0, u0)
#
# find optimal u0 such that
# x1 = xt  (for simple case xt = x0)

def nlp_multiple_shooting(eqs, N, dt):
    f = eqs['f']
    
    n_x = eqs['x'].numel()  # numbef of states
    n_u = eqs['u'].numel()  # number of inputs
    n_p = eqs['p'].numel()
    P = ca.SX.sym('P', n_p+(N+1)*n_x,1)
    p = P[:n_p]
    x0 = P[n_p:n_p+n_x]
    # xt = P[n_p+n_x:]
    
    Q_x = 5
    Q_y = 5
    Q_theta = 0.1
    R_v = 0.5
    R_rc = 0.005

    Q_x = 1000
    Q_y = 1000
    Q_theta = 1
    Q_vx = 0
    Q_vy = 0
    Q_omega = 0

    R1 = 1
    R2 = 1
    # Q = ca.diagcat(Q_x, Q_y, Q_theta, Q_vx, Q_vy, Q_omega)
    # R = ca.diagcat(R1, R2)

    Q = ca.diagcat(Q_x, Q_y, Q_theta)
    R = ca.diagcat(R_v, R_rc)

    t0 = 0

    x_opt = ca.SX.sym('x_opt', n_x, N+1)
    u_opt = ca.SX.sym('u_opt', n_u, N)
    
    # design vector for optimization
    xd_opt = ca.vertcat(x_opt.reshape((-1, 1)), u_opt.reshape((-1, 1)))
    
    f_cost = 0
    f_constraint = x_opt[:,0] - x0
    
    for k in range(N):
        u0 = u_opt[:,k]
        x = x_opt[:,k]
        xt = P[n_p+(k+1)*n_x:n_p+(k+2)*n_x]
        x_next = x_opt[:,k+1]
        h = ca.SX.sym('h')
        # one step of rk4
        
        k_1 = f(x, u0, p)
        k_2 = f(x + dt/2 * k_1, u0, p)
        k_3 = f(x + dt/2 * k_2, u0, p)
        k_4 = f(x + dt * k_3, u0, p)
        x1 = x + dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        # x1 = ca.substitute(rk4(f=lambda t, x: f(x0, u0, p), t=t0, y=x0, h=h), h, dt)
        
        # cost is how far we are at the end of the simulation from the desired target
        f_cost = f_cost + (x - xt).T @Q@ (x - xt) + u0.T@R@u0
        # print('f_cost', f_cost)
        f_constraint = ca.vertcat(f_constraint, x_next - x1) # how far dynamic simulation is off from rk4
    
    nlp_prob = {
        'f': f_cost,
        'x': xd_opt,
        'g': f_constraint,
        'p': P,
    }
    opts = {
        'ipopt':  {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6,
        },
        'print_time': 0,
    }
    return locals()

def update_param(x0, ref, k, N):
    p = ca.vertcat(x0)
    for l in range(N):
        if k+l < ref.shape[0]:
            ref_state = ref[k+l, :]
            v = 1
        else:
            ref_state = ref[-1, :]
            v = 0
        xt = ca.DM([ref_state[0], ref_state[1], ref_state[2]])
        p = ca.vertcat(p, xt)
    return p