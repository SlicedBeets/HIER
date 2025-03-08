import mujoco
import numpy as np

def skew(v):  # Cross-product matrix
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def compute_centroidal_momentum_matrix(model, data):

    #com_pos = data.subtree_com[1] # CoM Position 
    nv = model.nv
    A_G = np.zeros((6, nv))
    
    p_G = data.subtree_com[1].copy()
    
    for i in range(1, model.nbody):
        # Projection Matrix --------------------------------------
        p_i = data.xipos[i] # Get body CoM position in WORLD FRAME
        r = p_i - p_G  # Vector from global CoM to body CoM in inertial coordinates
        R = data.xmat[i].reshape(3,3) # Get rotation matrix: body to inertial coordinates
        S_r = skew(r) # Compute the skew-symmetric matrix S(r)
        # Spatial transform for motion vectors from CoM frame (G) to body frame (i): {}^i X_G
        iX_G = np.block([[R, np.zeros((3, 3))],
                         [R @ S_r.T, R]])
        # Spatial Inerita --------------------------------------
        mi = model.body_mass[i] # Mass of body i
        if mi == 0:
            # print(f"Body {i} has zero mass, skipping.") # <----- Uncomment to quickly Ident. 0 mass bodies
            continue
        com_inert = model.body_inertia[i] 
        I_cm = np.diag(com_inert, k=0) # Rotational Inerita for body (i) relative to CoM
        S_ci = skew(model.body_ipos[i]) # Skew of CoM position relative to body i coordinate frame
        I_bar_i = I_cm + mi*S_ci @ S_ci.T # Rotational Ineritia of Link i
        # Spatial Ineritia of Link i
        I_i = np.block([[I_bar_i, mi*S_ci],
                    [mi*S_ci.T, mi*np.eye(3)]])
        # Spatial Jacobian --------------------------------------
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacBodyCom(model, data, jacp, jacr, i)
        J_i = np.vstack([jacr, jacp])
        # Contribution to A_G
        A_G += iX_G.T @ I_i @ J_i
        
    return A_G


### <<<<<<<< WATCH OUT DEBUG HOT ZONE

# A_G = compute_centroidal_momentum_matrix(model, data)
# print(A_G)

### >>>>>>>> WATCH OUT DEBUG HOT ZONE