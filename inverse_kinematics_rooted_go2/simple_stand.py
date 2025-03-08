import numpy as np
import mujoco
import mujoco.viewer
import time
from dm_control import mjcf

# Load model from XML
xml_path = "/home/wrj/unitree_env/unitree_mujoco/unitree_robots/go2/scene.xml"
physics = mjcf.Physics.from_xml_path(xml_path)
model = physics.model
data = physics.data

# Define legs with joint names and actuators
feet = {
    "FL": {
        "site": "FL_foot_site",
        "joints": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "actuators": ["FL_hip", "FL_thigh", "FL_calf"]
    },
    "FR": {
        "site": "FR_foot_site",
        "joints": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "actuators": ["FR_hip", "FR_thigh", "FR_calf"]
    },
    "RL": {
        "site": "RL_foot_site",
        "joints": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "actuators": ["RL_hip", "RL_thigh", "RL_calf"]
    },
    "RR": {
        "site": "RR_foot_site",
        "joints": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "actuators": ["RR_hip", "RR_thigh", "RR_calf"]
    }
}

# Extract desired joint angles from the "home" keyframe
key_home = model.key('home')
q_des_all = key_home.qpos[7:19]  # Joint angles (12 values after base's 7 DOF)

# Set initial position to "home" pose
physics.data.qpos[:] = key_home.qpos
physics.forward()

# Dictionaries to store precomputed IDs
joint_ids = {}
actuator_ids = {}

# Precompute joint and actuator IDs for each leg
for leg, config in feet.items():
    joint_ids[leg] = [mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, j) 
                      for j in config["joints"]]
    actuator_ids[leg] = [mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, a) 
                         for a in config["actuators"]]

# Define PD controller gains
kp = 100.0  # Proportional gain
kd = 10.0   # Derivative gain

# Simulation loop with viewer
with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
    start_time = time.time()
    while viewer.is_running():
        # Compute control torques for each leg
        for leg in feet:
            for i, (joint_id, actuator_id) in enumerate(zip(joint_ids[leg], actuator_ids[leg])):
                # Get qpos and qvel indices for this joint
                qpos_index = physics.model.jnt_qposadr[joint_id]
                qvel_index = physics.model.jnt_dofadr[joint_id]
                
                # Desired, current position, and velocity
                q_des = q_des_all[qpos_index - 7]  # Offset by 7 due to free joint
                q = physics.data.qpos[qpos_index]
                qdot = physics.data.qvel[qvel_index]
                
                # PD control law: torque = kp * (q_des - q) - kd * qdot
                torque = kp * (q_des - q) - kd * qdot
                
                # Apply torque to actuator
                physics.data.ctrl[actuator_id] = torque

        # Advance the simulation
        physics.step()
        
        # Synchronize with the viewer
        viewer.sync()
        
        # Optional: Maintain real-time simulation
        elapsed = time.time() - start_time
        if elapsed < physics.timestep():
            time.sleep(physics.timestep() - elapsed)
        start_time = time.time()

print("Simulation ended.")