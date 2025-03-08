import numpy as np
import mujoco

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

# Constants
KP_COM = 2000.0
KD_COM = 50.0
KP_START = 100.0
KD_START = 10.0
KP_ROT = 500.0
KD_ROT = 20.0
DESIRED_COM_POS = np.array([0.0, 0.0, 0.35])
DESIRED_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
GRAVITY = np.array([0, 0, -9.81])

# Helper functions
def quaternion_error(q_des, q_curr):
    q_err = np.zeros(4)
    q_err[0] = q_des[0] * q_curr[0] + q_des[1] * q_curr[1] + q_des[2] * q_curr[2] + q_des[3] * q_curr[3]
    q_err[1] = q_des[0] * -q_curr[1] + q_des[1] * q_curr[0] + q_des[2] * -q_curr[3] + q_des[3] * q_curr[2]
    q_err[2] = q_des[0] * -q_curr[2] + q_des[1] * q_curr[3] + q_des[2] * q_curr[0] + q_des[3] * -q_curr[1]
    q_err[3] = q_des[0] * -q_curr[3] + q_des[1] * -q_curr[2] + q_des[2] * q_curr[1] + q_des[3] * q_curr[0]
    angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
    if abs(angle) < 1e-6:
        return np.zeros(3)
    return q_err[1:4] * (angle / np.linalg.norm(q_err[1:4] + 1e-6))

def print_force_block(current_time, settling_duration, com_pos, foot_forces, mode="Centroidal"):
    sys.stdout.write('\033[H')
    sys.stdout.write('\033[J')
    block = [
        f"Simulation Time: {current_time:.2f} seconds",
        f"Mode: {mode}",
        f"COM Position: x={com_pos[0]:.3f}, y={com_pos[1]:.3f}, z={com_pos[2]:.3f}",
        f"Desired COM z: {DESIRED_COM_POS[2]:.3f}",
        "Foot Forces (x, y, z):"
    ]
    for leg in FEET:
        force = foot_forces.get(leg, np.zeros(3))
        block.append(f"  {leg}: {force[0]:7.2f}, {force[1]:7.2f}, {force[2]:7.2f}")
    sys.stdout.write('\n'.join(block))
    sys.stdout.flush()

def read_input(input_queue):
    while True:
        command = input().strip().lower()
        input_queue.put(command)