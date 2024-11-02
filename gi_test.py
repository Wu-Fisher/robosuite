import xml.etree.ElementTree as ET
import gymnasium as  gym
from gymnasium import spaces
import numpy as np

from robosuite.models import MujocoWorldBase
from robosuite.models.arenas.table_arena import TableArena
from robosuite.models.grippers import RethinkGripper,PandaGripper
from robosuite.models.objects import BoxObject
from robosuite.utils import OpenCVRenderer
from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim
from robosuite.utils.mjcf_utils import new_actuator, new_joint

class GripperGymEnv(gym.Env):
    def __init__(self):
        super(GripperGymEnv, self).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # [x, y, z, gripper]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)  # Adjust as needed
        
        self.world = MujocoWorldBase()

        # Add a table
        arena = TableArena(table_full_size=(0.4, 0.4, 0.05), table_offset=(0, 0, 1.1), has_legs=False)
        self.world.merge(arena)

        # Add a gripper
        self.gripper = PandaGripper()
        gripper_body = ET.Element("body", name="gripper_base")
        gripper_body.set("pos", "0 0 1.3")
        gripper_body.set("quat", "0 0 1 0")  # flip z
        gripper_body.append(new_joint(name="gripper_x_joint", type="slide", axis="1 0 0", damping="50"))
        gripper_body.append(new_joint(name="gripper_y_joint", type="slide", axis="0 1 0", damping="50"))
        gripper_body.append(new_joint(name="gripper_z_joint", type="slide", axis="0 0 1", damping="50"))
        self.world.worldbody.append(gripper_body)
        self.world.merge(self.gripper, merge_body="gripper_base")
        self.world.actuator.append(new_actuator(joint="gripper_x_joint", act_type="position", name="gripper_x", kp="500"))
        self.world.actuator.append(new_actuator(joint="gripper_y_joint", act_type="position", name="gripper_y", kp="500"))
        self.world.actuator.append(new_actuator(joint="gripper_z_joint", act_type="position", name="gripper_z", kp="500"))

        # Add an object for grasping
        mujoco_object = BoxObject(name="box", size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 1], friction=[1, 0.005, 0.0001]).get_obj()
        mujoco_object.set("pos", "0 0 1.11")
        self.world.worldbody.append(mujoco_object)

        # Start simulation
        self.model = self.world.get_model(mode="mujoco")
        self.sim = MjSim(self.model)
        self.viewer = OpenCVRenderer(self.sim)
        self.render_context = MjRenderContextOffscreen(self.sim, device_id=-1)
        self.sim.add_render_context(self.render_context)

        self.sim_state = self.sim.get_state()
        self._ref_joint_vel_indexes = [self.sim.model.get_joint_qvel_addr(x) for x in ["gripper_x_joint", "gripper_y_joint", "gripper_z_joint"]]

    def reset(self):
        self.sim.set_state(self.sim_state)
        self.sim.step()
        observation = self._get_obs()
        return observation

    def step(self, action):
        x, y, z, gripper = action

        # Control gripper position
        self.sim.data.ctrl[self.sim.model.actuator_name2id("gripper_x")] = x
        self.sim.data.ctrl[self.sim.model.actuator_name2id("gripper_y")] = y
        self.sim.data.ctrl[self.sim.model.actuator_name2id("gripper_z")] = z
        
        # Control gripper open/close
        gripper_jaw_ids = [self.sim.model.actuator_name2id(x) for x in self.gripper.actuators]
        if gripper > 0:
            # self.sim.data.ctrl[gripper_jaw_ids] = [0.020833, -0.020833]  # Closed
            self.sim.data.ctrl[gripper_jaw_ids] = [0., -0.]  # Closed
        else:
            #self.sim.data.ctrl[gripper_jaw_ids] = [-0.0115, 0.0115]  # Open
            self.sim.data.ctrl[gripper_jaw_ids] = [0.04, -0.04]  # Open

        self.sim.step()
        self.sim.data.qfrc_applied[self._ref_joint_vel_indexes] = self.sim.data.qfrc_bias[self._ref_joint_vel_indexes]

        observation = self._get_obs()
        reward = self._compute_reward()
        done = self._is_done()

        return observation, reward, done, {}

    def _get_obs(self):
        # Implement observation retrieval
        obs = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.sim.data.ctrl.flat
        ])
        return obs

    def _compute_reward(self):
        # Implement reward computation
        reward = 0.0
        # Example: reward based on distance to object
        object_pos = self.sim.data.get_body_xpos("box_main")
        gripper_pos = self.sim.data.get_body_xpos("gripper_base")
        distance = np.linalg.norm(object_pos - gripper_pos)
        reward -= distance  # Penalize distance
        return reward

    def _is_done(self):
        # Implement termination condition
        # Example: task is done if the object is grasped and lifted
        return False

    def render(self, mode='human'):
        self.viewer.render()

# Example usage
if __name__ == "__main__":
    env = GripperGymEnv()
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break