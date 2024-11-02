from jrl.robots import Panda
# from jrl.evaluation import pose_errors_cm_deg
import torch

def assert_poses_almost_equal(poses_1, poses_2):
    # pos_errors_cm, rot_errors_deg = pose_errors_cm_deg(poses_1, poses_2)
    # assert (pos_errors_cm.max().item() < 0.01) and (rot_errors_deg.max().item() < 0.1)
    pass

robot = Panda()
joint_angles, poses = robot.sample_joint_angles_and_poses(n=5, return_torch=True) # sample 5 random joint angles and matching poses

print(f"sampled joint angles: {joint_angles}")
print(f"sampled poses: {poses}")


# Run forward-kinematics
poses_fk = robot.forward_kinematics(joint_angles) 
assert_poses_almost_equal(poses, poses_fk)

print(f"forward kinematics poses: {poses_fk}")

# Run inverse-kinematics
ik_sols = joint_angles + 0.1 * torch.randn_like(joint_angles) 
for i in range(5):
    ik_sols = robot.inverse_kinematics_step_levenburg_marquardt(poses, ik_sols)

print(f"ik_sols: {ik_sols}")

assert_poses_almost_equal(poses, robot.forward_kinematics(ik_sols))