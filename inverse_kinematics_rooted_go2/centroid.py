import numpy as np
import time
from dm_control import mjcf
import mujoco
import mujoco.viewer
import threading
import queue
import sys

# Load model from XML
xml_path = "/home/wrj/projects/inverse_kinematics_rooted_go2/scene_edit.xml"
physics = mjcf.Physics.from_xml_path(xml_path)
model = physics.model
data = physics.data

# Define legs with joint names and actuators (sites not used here but kept for reference)
feet = {
    "FL": {
        "bodies" : ["FL_hip", "FL_thigh", "FL_calf", "FL_foot"],
        "site": "FL_foot_site",
        "joints": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "actuators": ["FL_hip", "FL_thigh", "FL_calf"]
    },
    "FR": {
        "bodies" : ["FR_hip", "FR_thigh", "FR_calf", "FR_foot"],
        "site": "FR_foot_site",
        "joints": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "actuators": ["FR_hip", "FR_thigh", "FR_calf"]
    },
    "RL": {
        "bodies" : ["RL_hip", "RL_thigh", "RL_calf", "RL_foot"],
        "site": "RL_foot_site",
        "joints": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "actuators": ["RL_hip", "RL_thigh", "RL_calf"]
    },
    "RR": {
        "bodies" : ["RR_hip", "RR_thigh", "RR_calf", "RR_foot"],
        "site": "RR_foot_site",
        "joints": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "actuators": ["RR_hip", "RR_thigh", "RR_calf"],
    }
}

# Precomputing joint and actuator IDs
joint_ids = {}
actuator_ids = {}

for leg, config in feet.items():
    joint_ids[leg] = [model.joint(j).id for j in config["joints"]]
    actuator_ids[leg] = [model.actuator(a).id for a in config["actuators"]]
    config["body_ids"] = [mujoco.mj_name2id(physics.model.ptr, mujoco.mjtObj.mjOBJ_BODY, b) for b in config["bodies"]]
    print(f"{leg}: Body IDs = {config['body_ids']}")


# Extract Joint Angles and Position for Home Pose
key_home = model.key('home')
q_des_all = key_home.qpos[7:19]  # Joint angles (12 values after base's 7 DOF)

physics.data.qpos[:] = key_home.qpos
physics.forward()  # Update dependent quantities

kp_start = 100.0  # Startup PD gains
kd_start = 10.0   

kp_com = 2000.0  # Height Gains and Parameters
kd_com = 50.0   
desired_com_pos = np.array([0.0, 0.0, 0.385])
total_mass = sum(model.body_mass)
gravity = np.array([0, 0, -9.81])

kp_rot = 500   # Orientation Gains and Parameters
kd_rot = 20.0
desired_quat = np.array([1.0, 0.0, 0.0, 0.0])

def quaternion_error(q_des, q_curr): # Qunaternion Error Function
    # q_des * conj(q_curr)
    q_err = np.zeros(4)
    q_err[0] = q_des[0] * q_curr[0] + q_des[1] * q_curr[1] + q_des[2] * q_curr[2] + q_des[3] * q_curr[3]
    q_err[1] = q_des[0] * -q_curr[1] + q_des[1] * q_curr[0] + q_des[2] * -q_curr[3] + q_des[3] * q_curr[2]
    q_err[2] = q_des[0] * -q_curr[2] + q_des[1] * q_curr[3] + q_des[2] * q_curr[0] + q_des[3] * -q_curr[1]
    q_err[3] = q_des[0] * -q_curr[3] + q_des[1] * -q_curr[2] + q_des[2] * q_curr[1] + q_des[3] * q_curr[0]
    # Return vector part (x, y, z) scaled by angle
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if abs(angle) < 1e-6:
        return np.zeros(3)
    return q_err[1:4] * (angle / np.linalg.norm(q_err[1:4] + 1e-6))

def skew(v):  # Cross-product matrix
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


input_queue = queue.Queue()

def read_input(): # Function to read terminal input in a separate thread
    while True:
        command = input().strip().lower()
        input_queue.put(command)

# Start input thread
input_thread = threading.Thread(target=read_input, daemon=True)
input_thread.start()

def print_force_block(current_time, body_quat, quat_des, com_pos, foot_forces, mode="Centroidal"): # Function to print live force block
    # Move cursor to top-left and overwrite (ANSI escape code)
    sys.stdout.write('\033[H')  # Move cursor to top of terminal
    sys.stdout.write('\033[J')  # Clear from cursor down
    block = [
        f"Simulation Time: {current_time:.2f} seconds",
        f"Mode: {mode}",
        f"COM Position: x={com_pos[0]:.3f}, y={com_pos[1]:.3f}, z={com_pos[2]:.3f}",
        f"Desired COM z: {desired_com_pos[2]:.3f}",
        f"Rotation Quaternion : w={body_quat[0]:.3f}, i={body_quat[1]:.3f}, j={body_quat[2]:.3f}, k={body_quat[3]:.3f} ",
        f"Quat Error : w={quat_des[0]:.3f}, i={quat_des[1]:.3f}, j={quat_des[2]:.3f}, k={quat_des[3]:.3f} ",
        "Foot Forces (x, y, z):"
    ]
    for leg in feet:
        force = foot_forces.get(leg, np.zeros(3))  # Default to zero if not set
        block.append(f"  {leg}: {force[0]:7.2f}, {force[1]:7.2f}, {force[2]:7.2f}")
    sys.stdout.write('\n'.join(block))
    sys.stdout.flush()

with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer: # Simulation loop with viewer

    sim_start_time = time.time()
    settling_duration = 3.0 
    foot_forces = {}
    print("Type 'up' to increase COM height, 'down' to decrease, or 'quit' to exit.")

    while viewer.is_running():

        current_time = time.time() - sim_start_time
        
        try: # Check for terminal input
            command = input_queue.get_nowait()
            if command == 'up':
                desired_com_pos[2] += 0.01
                print(f"Desired COM height increased to {desired_com_pos[2]:.3f}")
            elif command == 'down':
                desired_com_pos[2] -= 0.01
                print(f"Desired COM height decreased to {desired_com_pos[2]:.3f}")
            elif command == 'quit':
                break
        except queue.Empty:
            pass


        if current_time < settling_duration: # Initial PD control for Each Leg
            # Compute control torques for each leg
            for leg in feet:
                for i, (joint_id, actuator_id) in enumerate(zip(joint_ids[leg], actuator_ids[leg])):
                    # Get qpos and qvel indices for this joint
                    qpos_index = physics.model.jnt_qposadr[joint_id]
                    qvel_index = physics.model.jnt_dofadr[joint_id]
                    body_quat = data.xquat[1]
                    # Desired, current position, and velocity
                    q_des = q_des_all[qpos_index - 7]  # Offset by 7 due to free joint
                    q = physics.data.qpos[qpos_index]
                    qdot = physics.data.qvel[qvel_index]
                    # PD control law: torque = kp * (q_des - q) - kd * qdot
                    torque = kp_start * (q_des - q) - kd_start * qdot
                    # Apply torque to actuator
                    physics.data.ctrl[actuator_id] = torque

            physics.step()
            viewer.sync()

            com_pos = data.subtree_com[0]
            print_force_block(current_time, body_quat, desired_quat, com_pos, foot_forces, mode="Settling (PD)")

       
        else:  # Centroidal Control
            
            com_pos = data.subtree_com[0] # CoM Position 
            com_vel = physics.data.cvel[0, 3:6] # CoM Velocity
            body_quat = data.xquat[1]  # Base_link quaternion
            body_angvel = physics.data.cvel[1, :3]  # Angular velocity
    
            # Position Control
            com_error = desired_com_pos - com_pos # [x, y, z] error
            com_vel_error = -com_vel
            total_force = kp_com * com_error + kd_com * com_vel_error + total_mass* gravity # Upward Force
            # COMMENTED OUT CLIPPING : total_force = np.clip(total_force, -2 * total_mass * abs(gravity[2]), 2 * total_mass * abs(gravity[2]))
            
            # Orientation control
            quat_error = quaternion_error(desired_quat, body_quat)
            total_torque = kp_rot * quat_error - kd_rot * body_angvel

            foot_forces = {}
            total_x_force = total_force[0]
            total_y_force = total_force[1]
            total_z_force = total_force[2]

            for leg in feet:
                foot_body_id = feet[leg]["body_ids"][-1]
                foot_pos = data.xpos[foot_body_id]
                foot_forces[leg] = np.array([
                    total_x_force / 4.0,
                    total_y_force / 4.0,
                    total_z_force / 4.0
                ])

                #Rotational Correction (cross product of position offset and torque)
                #r = foot_pos - com_pos
                #rot_force = np.cross(r, total_torque / 4.0)  # Distribute torque evenly
                #foot_forces[leg] += rot_force
            
            # Compute control torques for each actuator
            for leg in feet:
                # Get foot body ID from config
                foot_body_id = feet[leg]["body_ids"][-1]
                foot_pos = physics.data.xpos[foot_body_id]

                # Compute Jacobian for foot
                jacp = np.zeros((3, model.nv)) # Position J
                mujoco.mj_jacBody(model.ptr, data.ptr, jacp, None, foot_body_id)
                jnt_dof_ids = [model.jnt_dofadr[jid] for jid in joint_ids[leg]]
                J = jacp[:, jnt_dof_ids] # 3x3 Jacobian
                torques = -J.T @ foot_forces[leg]

                # Apply Torques to Actuators
                for i, actuator_id in enumerate(actuator_ids[leg]):
                    data.ctrl[actuator_id] = torques[i]

            physics.step()
            print_force_block(current_time, body_quat, desired_quat, com_pos, foot_forces)
            viewer.sync()

            # Maintain real-time simulation
            step_start_time = time.time()
            elapsed = time.time() - step_start_time
            if elapsed < physics.timestep():
                time.sleep(physics.timestep() - elapsed)


print("\nSimulation ended.")