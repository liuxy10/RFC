import mujoco_py
import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append(os.getcwd())
from khrylib.rl.envs.common import mujoco_env

if __name__ == "__main__":    

    env =  mujoco_env.MujocoEnv('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/temp_feet.xml', 1)
    # Load the model and create a simulator
    PATH_TO_XML = os.path.expanduser('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/temp_feet.xml')
    model = mujoco_py.load_model_from_path(PATH_TO_XML)
    sim = mujoco_py.MjSim(model)

    # Set the time interval for the simulator
    sim.model.opt.timestep = 0.01  # Set the time step to 0.01 seconds

    # Set the gravity constant
    sim.model.opt.gravity[:] = [0, 0, -9.81]  # Set gravity to -9.81 m/s^2 in the z-axis

    # Set the initial position and velocity
    sim.data.qpos[:] = [0, 0, 1, 0, 0, 0, 0]  # Set initial position to (0, 0, 1) to observe falling
    sim.data.qvel[:] = np.zeros(6)  # Set initial velocity to (0, 0, 0)

    # Simulate and collect contact forces
    contact_forces = []
    root_body_positions = []
    
    sim.forward()
    for _ in range(100000):  # Simulate 1000 steps
        sim.step()
        root_body_pos = sim.data.qpos[:3]  # Assuming the root body position is the first three elements in qpos
        root_body_positions.append(root_body_pos)
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            geom2_body = sim.model.geom_bodyid[contact.geom2]
            force = sim.data.cfrc_ext[geom2_body]
            contact_forces.append(force)
            # Get the position of the root body

            
        

    # Visualize the root body positions
    root_body_positions = np.array(root_body_positions)
    plt.figure(figsize=(10, 6))
    plt.plot(root_body_positions[:, 0], label='Position X')
    plt.plot(root_body_positions[:, 1], label='Position Y')
    plt.plot(root_body_positions[:, 2], label='Position Z')
    plt.title('Root Body Positions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.show()


    # Visualize the contact forces
    contact_forces = np.array(contact_forces)
    plt.figure(figsize=(10, 6))
    plt.plot(contact_forces[:, 0], label='Force X')
    plt.plot(contact_forces[:, 1], label='Force Y')
    plt.plot(contact_forces[:, 2], label='Force Z')
    plt.title('Contact Forces Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Force Magnitude')
    plt.legend()
    plt.show()
