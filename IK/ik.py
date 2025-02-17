import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class InvertedPendulum:
    def __init__(self, L=1.0):
        self.L = L  # Length of pendulum
        
    def forward_kinematics(self, q, marker_local):
        """
        Forward kinematics for inverted pendulum
        q: angle from vertical
        marker_local: local coordinates of marker in pendulum frame
        """
        # Rotation matrix
        R = np.array([[np.cos(q), -np.sin(q)],
                     [np.sin(q), np.cos(q)]])
        
        # Origin of pendulum (fixed at 0,0)
        p0 = np.array([0.0, 0.0])
        
        # Transform marker from local to world coordinates
        marker_world = p0 + R @ marker_local
        return marker_world


def objective_function(q_sequence, pendulum, marker_local, target_positions):
    """
    Implements Equation 1 from the paper:
    min_{q_{1:T}, s, p} sum_{t=1}^T sum_{i=1}^M ||f_FK(q_t, s, p^(i)) - x_t^(i)||
    
    Simplified for single marker inverted pendulum case
    """
    total_error = 0
    
    for t in range(len(q_sequence)):
        # Get marker position through forward kinematics
        predicted_pos = pendulum.forward_kinematics(q_sequence[t], marker_local)
        
        # Calculate error between predicted and target position
        error = np.linalg.norm(predicted_pos - target_positions[t])
        total_error += error
        
    return total_error

# # Example usage
# if __name__ == "__main__":
#     # Create pendulum
#     pendulum = InvertedPendulum(L=1.0)
    
#     # Define marker in local coordinates (at end of pendulum)
#     marker_local = np.array([0.0, 1.0])
    
#     # Generate synthetic target data (simple circular motion)
#     t = np.linspace(0, 2*np.pi, 100)
#     true_angles = 0.5 * np.sin(t)  # True angle sequence
#     target_positions = np.array([pendulum.forward_kinematics(q, marker_local) 
#                                for q in true_angles]) * 1.1  # Add some error
    
#     # Initial guess for optimization
#     q_init = np.zeros_like(true_angles)
    
#     print("Optimize to find joint angles")
#     result = minimize(objective_function, q_init, 
#                      args=(pendulum, marker_local, target_positions),
#                      method='BFGS')
    
#     optimized_angles = result.x
    
#     # Plot results
#     plt.figure(figsize=(12, 4))
#     plt.subplot(121)
#     plt.plot(t, true_angles, 'b-', label='True angles')
#     plt.plot(t, optimized_angles, 'r--', label='Optimized angles')
#     plt.legend()
#     plt.title('Joint Angles')
    
#     plt.subplot(122)
#     plt.plot(target_positions[:,0], target_positions[:,1], 'b-', label='Target trajectory')
#     optimized_positions = np.array([pendulum.forward_kinematics(q, marker_local) 
#                                   for q in optimized_angles])
#     plt.plot(optimized_positions[:,0], optimized_positions[:,1], 'r--', 
#              label='Optimized trajectory')
#     plt.legend()
#     plt.title('Marker Trajectories')
#     plt.axis('equal')
#     plt.show()

import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

class TwoLinkPendulum:
    def __init__(self, link_lengths=[1.0, 1.0]):
        # State description:
        # q = [x_base, z_base, q1, q2]
        # where:
        # x_base, z_base: planar base position
        # q1: angle of first link relative to vertical (positive counterclockwise)
        # q2: angle of second link relative to first link (positive counterclockwise)
        self.l1, self.l2 = link_lengths
        self.dof = 4  # degrees of freedom [x_base, z_base, q1, q2]
        
    def forward_kinematics(self, q, marker_local):
        """
        Forward kinematics for marker positions
        q: [x_base, z_base, q1, q2]
        marker_local: [link_idx, x_local, z_local]
        """
        x_base, z_base, q1, q2 = q
        link_idx, x_local, z_local = marker_local
        
        # Rotation matrices
        R1 = np.array([[np.cos(q1), -np.sin(q1)],
                      [np.sin(q1), np.cos(q1)]])
        R2 = np.array([[np.cos(q1 + q2), -np.sin(q1 + q2)],
                      [np.sin(q1 + q2), np.cos(q1 + q2)]])
        
        # Base position
        p0 = np.array([x_base, z_base])
        
        if link_idx == 0:  # First link
            pos = p0 + R1 @ np.array([x_local, z_local])
        else:  # Second link
            p1 = p0 + R1 @ np.array([0, self.l1])  # First joint position
            pos = p1 + R2 @ np.array([x_local, z_local])
            
        return pos
    
    def compute_jacobian(self, q, marker_local):
        """
        Compute Jacobian matrix for marker position
        Returns 2x4 matrix: [dx/dq, dz/dq]
        """
        x_base, z_base, q1, q2 = q
        link_idx, x_local, z_local = marker_local
        
        J = np.zeros((2, 4))
        
        # Base derivatives
        J[:, 0:2] = np.eye(2)
        
        if link_idx == 0:  # First link
            # Derivative w.r.t q1
            J[0, 2] = -x_local * np.sin(q1) - z_local * np.cos(q1)
            J[1, 2] = x_local * np.cos(q1) - z_local * np.sin(q1)
            # No q2 dependence
            J[:, 3] = 0
            
        else:  # Second link
            # First joint position
            p1 = np.array([x_base, z_base]) + \
                 np.array([self.l1 * np.sin(q1), self.l1 * np.cos(q1)])
            
            # Derivative w.r.t q1
            J[0, 2] = -x_local * np.sin(q1 + q2) - z_local * np.cos(q1 + q2) + \
                      self.l1 * np.cos(q1)
            J[1, 2] = x_local * np.cos(q1 + q2) - z_local * np.sin(q1 + q2) - \
                      self.l1 * np.sin(q1)
            
            # Derivative w.r.t q2
            J[0, 3] = -x_local * np.sin(q1 + q2) - z_local * np.cos(q1 + q2)
            J[1, 3] = x_local * np.cos(q1 + q2) - z_local * np.sin(q1 + q2)
            
        return J

def solve_ik(pendulum, target_pos, marker_local, initial_q=None, 
             max_iter=100, tol=1e-3):
    """
    Solve inverse kinematics using Jacobian pseudo-inverse method
    """
    if initial_q is None:
        initial_q = np.zeros(pendulum.dof)
        
    q = initial_q.copy()
    
    for i in range(max_iter):
        current_pos = pendulum.forward_kinematics(q, marker_local)
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < tol:
            break
            
        J = pendulum.compute_jacobian(q, marker_local)
        delta_q = pinv(J) @ error
        q += delta_q
        
    return q

# Example usage
if __name__ == "__main__":
    # Create pendulum
    pendulum = TwoLinkPendulum(link_lengths=[1.0, 1.0])
    
    # Define marker on second link end
    marker_local = [1, 0.0, 1.0]  # [link_idx, x_local, z_local]
    
    # Target position
    target_pos = np.array([0.5, 1.5])
    
    # Solve IK
    initial_q = np.array([0.0, 0.0, 0.1, 0.1])
    q_sol = solve_ik(pendulum, target_pos, marker_local, initial_q)
    
    print(f"Solution q: {q_sol}")
