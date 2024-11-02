"""
This script shows how to adapt an environment to be compatible
with the Gymnasium API. This is useful when using
learning pipelines that require supporting these APIs.

For instance, this can be used with OpenAI Baselines
(https://github.com/openai/baselines) to train agents
with RL.


We base this script off of some code snippets found
in the "Basic Usage" section of the Gymnasium documentation

The following snippet was used to demo basic functionality.

    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            env.close()

To adapt our APIs to be compatible with OpenAI Gym's style, this script
demonstrates how this can be easily achieved by using the GymWrapper.
"""

import robosuite as suite
from robosuite.wrappers import GymWrapper,VisualizationWrapper

# from robosuite import  load_part_controller_config
# 创建RoboSuite环境并包装成Gym环境
import numpy as np


# controller_config = load_controller_config(default_controller="IK_POSE")

# controller_config = load_part_controller_config(default_controller="IK_POSE")

if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(VisualizationWrapper(
        suite.make(
            "MyLift",
            robots="Panda",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
            # controller_configs=controller_config
        ))
    )

    # print("env.action_space", env.action_space)
    # print("env.observation_space", env.observation_space)
    # accel_left_finger_tip_id = env.sim.model.sensor_name2id('gripper0_accel_left_finger_tip')
    # accel_right_finger_tip_id = env.sim.model.sensor_name2id('gripper0_accel_right_finger_tip')

    env.reset(seed=1)
    import time
    for i_episode in range(3):
        observation = env.reset()
        env._ee_close_to_cube()
        # env.sim.data.qvel[:] = 0.
        # sim = env.sim
        # for body_id in range(sim.model.nbody):
        #     print(sim.model.body_id2name(body_id))
        #     body_pos = sim.data.body_xpos[body_id]
        #     print(f"body_pos: {body_pos}")

        # for joint_id in range(sim.model.njnt):
        #     print(sim.model.joint_id2name(joint_id))
        #     joint_pos = sim.data.qpos[joint_id]
        #     print(f"joint_pos: {joint_pos}")
        #     joint_vel = sim.data.qvel[joint_id]
        #     print(f"joint_vel: {joint_vel}")


        # target_pos = sim.data.body_xpos[env.cube_body_id]
        # print(f"target_pos: {target_pos}")
        # target_quat = sim.data.body_xquat[env.cube_body_id]
        # print(f"target_quat: {target_quat}")

        # print(f"action_space: {env.action_space}")
        # exit()
        # print(type(env.robots[0].composite_controller.))
        # print(dir(env.robots[0].composite_controller))
        # print([part for part in env.robots[0].composite_controller.part_controllers])
        # print([type(c) for c in env.robots[0].composite_controller.part_controllers.values()])
        # exit()

        action =  np.zeros_like(env.action_space.sample())
        for t in range(500):
            env.render()
            # action = env.action_space.sample()
            # action =  np.zeros_like(env.action_space.sample())
            # action = np.zeros_like(env.action_space.sample())
            # action[2] = 0.01
            observation, reward, terminated, truncated, info = env.step(action)
            # print(f"reward: {reward}")
            # print(f"joint_positions: {env.sim.data.qpos[env.robots[0].joint_indexes]}")
            # print(f"joint_velocities: {env.sim.data.qvel[env.robots[0].joint_indexes]}")
            # env.sim.data.qvel[:] = 0.
            # print(f"left_finger_tip_accel: {env.sim.data.sensordata[accel_left_finger_tip_id:accel_left_finger_tip_id+3]}")
            # print(f"right_finger_tip_accel: {env.sim.data.sensordata[accel_right_finger_tip_id:accel_right_finger_tip_id+3]}")
            
            # time.sleep(1)
            
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                env.close()
                break