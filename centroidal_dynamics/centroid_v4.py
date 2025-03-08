import numpy as np
import mujoco.viewer
from dm_control import mujoco, mjcf
from Centroidal_MM_V1 import compute_centroidal_momentum_matrix, skew
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import time
import threading
import queue
import sys

# --- Utility Functions ---

def quaternion_error(q_des, q_curr):
    """Compute the quaternion error between desired and current quaternions."""
    q_err = np.zeros(4)
    q_err[0] = q_des[0] * q_curr[0] + q_des[1] * q_curr[1] + q_des[2] * q_curr[2] + q_des[3] * q_curr[3]
    q_err[1] = q_des[0] * -q_curr[1] + q_des[1] * q_curr[0] + q_des[2] * -q_curr[3] + q_des[3] * q_curr[2]
    q_err[2] = q_des[0] * -q_curr[2] + q_des[1] * q_curr[3] + q_des[2] * q_curr[0] + q_des[3] * -q_curr[1]
    q_err[3] = q_des[0] * -q_curr[3] + q_des[1] * -q_curr[2] + q_des[2] * q_curr[1] + q_des[3] * q_curr[0]
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if abs(angle) < 1e-6:
        return np.zeros(3)
    return q_err[1:4] * (angle / np.linalg.norm(q_err[1:4] + 1e-6))

def read_input(input_queue):
    """Read terminal input in a separate thread and put commands into a queue."""
    while True:
        command = input().strip().lower()
        input_queue.put(command)

def print_block(current_time, body_quat, quat_des, com_pos, foot_forces, Ag_mine, Ag_pin, model, data, h_G_true=0, h_G_pred_mine=0, h_G_pred_pin=0, mode="Centroidal"):
    """Print simulation status including forces and momentum matrices."""
    sys.stdout.write('\033[H')  # Move cursor to top
    sys.stdout.write('\033[J')  # Clear screen
    block = [
        f"Simulation Time: {current_time:.2f} seconds",
        f"Mode: {mode}",
        f"COM Position: x={com_pos[0]:.3f}, y={com_pos[1]:.3f}, z={com_pos[2]:.3f}",
        f"Desired COM z: {desired_com_pos[2]-com_pos[2]:.3f}",
        f"Rotation Quaternion: w={body_quat[0]:.3f}, i={body_quat[1]:.3f}, j={body_quat[2]:.3f}, k={body_quat[3]:.3f}",
        f"Quat Error: w={quat_des[0]-body_quat[0]:.3f}, i={quat_des[1]-body_quat[1]:.3f}, j={quat_des[2]-body_quat[2]:.3f}, k={quat_des[3]-body_quat[3]:.3f}",
        "Foot Forces (x, y, z):"
    ]
    for leg in feet:
        force = foot_forces.get(leg, np.zeros(3))
        block.append(f"  {leg}: {force[0]:7.2f}, {force[1]:7.2f}, {force[2]:7.2f}")
    block.append("My Centroidal Momentum Matrix:")
    for i in range(6):
        row_str = ', '.join([f"{Ag_mine[i,j]:.3f}" for j in range(Ag_mine.shape[1])])
        block.append(f"  Row {i}: [{row_str}]")
    block.append("Pinocchio Centroidal Momentum Matrix:")
    for i in range(6):
        row_str = ', '.join([f"{Ag_pin[i,j]:.3f}" for j in range(Ag_pin.shape[1])])
        block.append(f"  Row {i}: [{row_str}]")

    sys.stdout.write('\n'.join(block))
    sys.stdout.flush()

def debug_block(current_time, com_pos, Ag_mine, Ag_pin, model, data, mode="Centroidal"):
    sys.stdout.write('\033[H')  # Move cursor to top
    sys.stdout.write('\033[J')  # Clear screen
    block = [
        f"Simulation Time: {current_time:.2f} seconds",
        f"Mode: {mode}",
        f"COM Position: x={com_pos[0]:.3f}, y={com_pos[1]:.3f}, z={com_pos[2]:.3f}",
    ]
    block.append(f"Data CVEL :")
    for i in range(model.nbody):
        row_str = ', '.join([f"{data.cvel[i,j]:.3f}" for j in range(data.cvel.shape[1])])
        block.append(f"  Row {i}: [{row_str}]")
    block.append(f"Data linvel :")
    for i in range(model.nbody):
        row_str = ', '.join([f"{data.subtree_linvel[i,j]:.3f}" for j in range(data.subtree_linvel.shape[1])])
        block.append(f"  Row {i}: [{row_str}]")
    sys.stdout.write('\n'.join(block))
    sys.stdout.flush()
    return

# --- Control Functions ---

def apply_pd_control(model, data, feet, q_des_all, kp, kd):
    """Apply PD control to each leg's joints during the settling phase."""
    for leg in feet:
        for i, (joint_id, actuator_id) in enumerate(zip(feet[leg]["joint_ids"], feet[leg]["actuator_ids"])):
            qpos_index = model.jnt_qposadr[joint_id]
            qvel_index = model.jnt_dofadr[joint_id]
            q_des = q_des_all[qpos_index - 7]
            q = data.qpos[qpos_index]
            qdot = data.qvel[qvel_index]
            torque = kp * (q_des - q) - kd * qdot
            data.ctrl[actuator_id] = torque

def compute_desired_wrench(com_pos, com_vel, body_quat, body_angvel, desired_com_pos, desired_quat, total_mass, gravity, kp_com, kd_com, kp_rot, kd_rot):
    """Compute the desired wrench for centroidal control."""
    com_error = desired_com_pos - com_pos
    com_vel_error = -com_vel
    total_force = kp_com * com_error + kd_com * com_vel_error + total_mass * gravity
    quat_error = quaternion_error(desired_quat, body_quat)
    total_torque = kp_rot * quat_error - kd_rot * body_angvel
    return np.concatenate([total_force, total_torque])

def compute_foot_forces(model, data, feet, com_pos, W_body):
    """Compute foot forces using the grasp matrix for centroidal control."""
    A = np.zeros((6, 3 * len(feet)))
    for i, leg in enumerate(feet):
        site_id = feet[leg]["site_id"]
        foot_pos = data.site_xpos[site_id]
        r = foot_pos - com_pos
        A[0:3, 3*i:3*i+3] = np.eye(3)
        A[3:6, 3*i:3*i+3] = skew(r)
    F = np.linalg.pinv(A) @ (-W_body)
    foot_forces = {leg: F[3*i:3*i+3] for i, leg in enumerate(feet)}
    return foot_forces

def apply_foot_forces(model, data, feet, foot_forces):
    """Apply computed foot forces to actuators via joint torques."""
    for leg in feet:
        site_id = feet[leg]["site_id"]
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model.ptr, data.ptr, jacp, None, site_id)
        jnt_dof_ids = [model.jnt_dofadr[jid] for jid in feet[leg]["joint_ids"]]
        J = jacp[:, jnt_dof_ids]
        torques = J.T @ foot_forces[leg]
        for i, actuator_id in enumerate(feet[leg]["actuator_ids"]):
            data.ctrl[actuator_id] = torques[i]

# --- Dynamics Functions ---

def compute_ag_mujoco(model, data):
    """Compute the centroidal momentum matrix using MuJoCo."""
    return compute_centroidal_momentum_matrix(model.ptr, data.ptr)

def compute_ag_pinocchio(pin_model, pin_data, q_pin, v_pin):
    """Compute the centroidal momentum matrix using Pinocchio."""
    pin.forwardKinematics(pin_model, pin_data, q_pin, v_pin)
    pin.computeCentroidalMap(pin_model, pin_data, q_pin)
    return pin_data.Ag

def pinocchio_viz(viz, q):
    
    viz.display(q)
    
    return

# --- Main Script ---

# Mujoco Model
xml_path = "/home/wrj/projects/centroidal_dynamics/scene_edit.xml"
physics = mjcf.Physics.from_xml_path(xml_path)
model = physics.model
data = physics.data

# Pinocchio Model and Attempt at Meshcat Viewer (Does not work on WSL)
pin_model, collision_model, visual_model = pin.buildModelsFromMJCF(
    "/home/wrj/projects/centroidal_dynamics/go2_edit.xml",
    geometry_types=[pin.GeometryType.COLLISION, pin.GeometryType.VISUAL],)
pin_data = pin_model.createData()

# Define robot legs
feet = {
    "FL": {
        "bodies": ["FL_hip", "FL_thigh", "FL_calf", "FL_foot"],
        "site": "FL_foot_site",
        "joints": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        "actuators": ["FL_hip", "FL_thigh", "FL_calf"]
    },
    "FR": {
        "bodies": ["FR_hip", "FR_thigh", "FR_calf", "FR_foot"],
        "site": "FR_foot_site",
        "joints": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        "actuators": ["FR_hip", "FR_thigh", "FR_calf"]
    },
    "RL": {
        "bodies": ["RL_hip", "RL_thigh", "RL_calf", "RL_foot"],
        "site": "RL_foot_site",
        "joints": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        "actuators": ["RL_hip", "RL_thigh", "RL_calf"]
    },
    "RR": {
        "bodies": ["RR_hip", "RR_thigh", "RR_calf", "RR_foot"],
        "site": "RR_foot_site",
        "joints": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        "actuators": ["RR_hip", "RR_thigh", "RR_calf"]
    }
}

# Precompute IDs
for leg in feet:
    feet[leg]["body_ids"] = [mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_BODY, b) for b in feet[leg]["bodies"]]
    feet[leg]["site_id"] = mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_SITE, feet[leg]["site"])
    feet[leg]["joint_ids"] = [mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_JOINT, j) for j in feet[leg]["joints"]]
    feet[leg]["actuator_ids"] = [mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in feet[leg]["actuators"]]

# Set initial pose from keyframe
key_home = model.key('home')
q_des_all = key_home.qpos[7:19]
data.qpos[:] = key_home.qpos
physics.step()

# Define constants
kp_start = 100.0
kd_start = 10.0
kp_com = 2000.0
kd_com = 1
desired_com_pos = np.array([0.0, 0.0, 0.385])
total_mass = sum(model.body_mass)
gravity = np.array([0, 0, -9.81])
kp_rot = 500.0
kd_rot = 20.0
desired_quat = np.array([1.0, 0.0, 0.0, 0.0])
com_site_id = mujoco.mj_name2id(model.ptr, mujoco.mjtObj.mjOBJ_SITE, "com_marker")

# Start input thread
input_queue = queue.Queue()
input_thread = threading.Thread(target=read_input, args=(input_queue,), daemon=True)
input_thread.start()

# Simulation loop
with mujoco.viewer.launch_passive(model.ptr, data.ptr) as viewer:
    sim_start_time = time.time()
    settling_duration = 3.0
    foot_forces = {}
    print("Type 'up' to increase COM height, 'down' to decrease, or 'quit' to exit.")

    while viewer.is_running():
        current_time = time.time() - sim_start_time

        # Handle user input
        try:
            command = input_queue.get_nowait()
            if command == 'up':
                desired_com_pos[2] += 0.025
                print(f"Desired COM height increased to {desired_com_pos[2]:.3f}")
            elif command == 'down':
                desired_com_pos[2] -= 0.025
                print(f"Desired COM height decreased to {desired_com_pos[2]:.3f}")
            elif command == 'quit':
                break
        except queue.Empty:
            pass

        if current_time < settling_duration: # Settling phase with PD control
            apply_pd_control(model, data, feet, q_des_all, kp_start, kd_start)

            physics.step()

            Ag_mine = compute_ag_mujoco(model, data)
            q_pin = data.qpos.copy()
            v_pin = data.qvel.copy()
            Ag_pin = compute_ag_pinocchio(pin_model, pin_data, q_pin, v_pin)
            com_pos = data.subtree_com[1]
            body_quat = data.xquat[1]
            print_block(current_time, body_quat, desired_quat, com_pos, foot_forces, Ag_mine, Ag_pin, model, data, mode="Settling (PD)")

        else: # Centroidal control phase
            
            com_pos = data.subtree_com[1]
            com_vel = data.cvel[1, 3:6]
            body_quat = data.xquat[1]
            body_angvel = data.cvel[1, :3]
            W_body = compute_desired_wrench(com_pos, com_vel, body_quat, body_angvel, desired_com_pos, desired_quat, total_mass, gravity, kp_com, kd_com, kp_rot, kd_rot)
            foot_forces = compute_foot_forces(model, data, feet, com_pos, W_body)
            apply_foot_forces(model, data, feet, foot_forces)

            physics.step()

            Ag_mine = compute_ag_mujoco(model, data)
            q_pin = data.qpos.copy()
            v_pin = data.qvel.copy()
            Ag_pin = compute_ag_pinocchio(pin_model, pin_data, q_pin, v_pin)
            data.site_xpos[com_site_id] = data.subtree_com[1]

            m_total = model.body_subtreemass[1]  # Total mass of all bodies (skip world body)
            v_G = data.subtree_linvel[1].copy()  # Velocity of the CoM
            l = m_total * v_G
        
            #k_G = data.cvel[3:6]  # Angular momentum

            h_G_true = l
            h_G_pred_mine = Ag_mine @ data.qvel
            h_G_pred_pin = Ag_pin @ data.qvel

            print_block(current_time, body_quat, desired_quat, com_pos, foot_forces, Ag_mine, Ag_pin, model, data)
            # debug_block(current_time, com_pos, Ag_mine, Ag_pin, model, data)

        viewer.sync()
        elapsed = time.time() - (time.time() - sim_start_time + current_time)
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

print("\nSimulation ended.")