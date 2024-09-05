#!/usr/bin/env cyecca_python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Joy
from tf_transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation
import casadi as ca
import numpy as np
from corti.bezier_rover_planning import generate_path
from corti.rover_planning import RoverPlanner
from NightVaporMPC import *

v  = 1
r = 1
planner = RoverPlanner(x=0, y=0, v=v, theta=0, r=r)
planner.goto(3, 0, v, r)
# planner.goto(3, 6, v, r)
# planner.goto(0, 3, v, r)
# planner.goto(0, 0, v, r)
# planner.goto(-3, 0, v, r)
# planner.goto(-3, 3, v, r)
# planner.goto(0, 3, v, r)
# planner.goto(0, 6, v, r)
# planner.goto(3, 6, v, r)
# planner.goto(3, 3, v, r)
# planner.goto(-3, 3, v, r)
# planner.goto(-3, 6, v, r)
planner.stop(3, 0)

ref_data = planner.compute_ref_data(plot=True)
t = ref_data['t']
p = ca.vertcat(1)

xt = np.array([ref_data['x'](t), ref_data['y'](t), ref_data['theta'](t)]).T

eqs = derive_dynamics()

N = 15
dt = 0.1

p = ca.vertcat(1)

nlp = nlp_multiple_shooting(eqs,N,dt)
solver = ca.nlpsol('solver', 'ipopt', nlp['nlp_prob'], nlp['opts'])

n_x = nlp['n_x']
n_u = nlp['n_u']

v_max = 1
r_c_max = 1
v_min = -1
r_c_min = -r_c_max
lbg = ca.DM.zeros((n_x * (N+1)))
ubg = -ca.DM.zeros((n_x * (N+1)))

lbx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))
ubx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))

# states x: [w, r, vx, vy, omega, px, py, theta]
lbx[0:n_x * (N + 1):n_x] = -ca.inf
lbx[1:n_x * (N + 1):n_x] = -ca.inf
lbx[2:n_x * (N + 1):n_x] = -ca.inf

ubx[0:n_x * (N + 1):n_x] = ca.inf
ubx[1:n_x * (N + 1):n_x] = ca.inf
ubx[2:n_x * (N + 1):n_x] = ca.inf

lbx[n_x*(N + 1)::n_u] = v_min
ubx[n_x*(N + 1)::n_u] = v_max
lbx[n_x*(N + 1)+1::n_u] = r_c_min
ubx[n_x*(N + 1)+1::n_u] = r_c_max



args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx,
}

class NightVaporPublisher(Node):
    def __init__(self):
        super().__init__('night_vapor_publisher')
        self.pub_control_input = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_mocap = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_cb, 10)
        self.timer = self.create_timer(dt, self.pub_night_vapor)
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.u0 = ca.DM.zeros((n_u,N))
        self.i = 0

    def pose_cb(self, msg: PoseStamped):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.theta = np.deg2rad(yaw)

    def pub_night_vapor(self):
        state_0 = np.array([self.x, self.y, self.theta])
        X0 = ca.repmat(state_0, 1, N + 1)

        if ca.norm_2(state_0 - xt[self.i]) > 1e-1:
            print(ca.norm_2(state_0 - xt[self.i]))
            # print('mpc')
            print(state_0)
            print(xt[self.i])
            args['P'] = ca.vertcat(p, update_param(state_0, xt, self.i, N))

            args['x0'] = ca.vertcat(ca.reshape(X0, n_x * (N + 1), 1),
                                    ca.reshape(self.u0, n_u * N, 1))

            sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['P'])
            u = ca.reshape(sol['x'][n_x * (N + 1):], n_u, N)
            cmd_vel = u[:,0]
            # print(self.i, cmd_vel)

            vel_msg = Twist()
            vel_msg.linear.x = float(cmd_vel[0])
            vel_msg.angular.z = float(cmd_vel[0])*ca.tan(float(cmd_vel[1]))
            self.pub_control_input.publish(vel_msg)
            X = ca.reshape(sol['x'][:n_x * (N + 1)], n_x, (N + 1))
            
            self.u0 = ca.horzcat(u[:, 1:], ca.reshape(u[:, -1], -1, 1))
            X0 = ca.horzcat(X[:, 1:],ca.reshape(X[:, -1], -1, 1))
        else:
            self.i = self.i + 1
            print('update', self.i)
    

def main(args=None):
    rclpy.init(args=args)
    night_vapor_publisher = NightVaporPublisher()
    rclpy.spin(night_vapor_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    night_vapor_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
