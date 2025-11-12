import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Optional
import genesis as gs
import torch
import math
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch.nn as nn
from typing import Optional
from scipy.spatial.transform import Rotation as R

def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

class PiperEnv(gym.Env):
  
    def __init__(self, visualize: bool = False):
        super(PiperEnv, self).__init__()

        self.visualize = visualize
        self.kp = torch.tensor([4500, 4500, 3500, 3500, 2500.0, 2500.0], device="cpu")
        self.kv = torch.tensor([450.0, 450.0, 350.0, 350.0, 250.0, 250.0], device="cpu")
        self.jnt_name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6"
        ]
        self.workspace = {
            'x': [-0.5, 1.5],
            'y': [-0.8, 0.8],
            'z': [0.05, 0.5]
        }
        self.tensor_device = "cpu"
        self.jnt_range = torch.tensor([
            [-2.61, 2.61],
            [0, 3.14],
            [-2.7, 0],
            [-1.83, 1.83],
            [-1.22, 1.22],
            [-1.57, 1.57]
        ], device=self.tensor_device)

        self.gs_device = gs.cpu
        gs.init(backend = self.gs_device)
        self.scene = gs.Scene(
            show_viewer = self.visualize,
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (3.5, -1.0, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 40,
            ),
            rigid_options = gs.options.RigidOptions(
                dt = 0.01,
            ),
        )

        plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file='xml/agilex_piper/piper.xml'),
        )

        self.scene.build()

        self.default_joint_pos = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.tensor_device)
        self.default_ee_pos = torch.tensor([0.0, 0.0, 0.0], device=self.tensor_device)

        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.jnt_name]
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kv, self.motors_dof_idx)

        self.goal = torch.tensor(torch.zeros(3, dtype=torch.float32), device=self.tensor_device)
        self.last_action = torch.tensor(torch.zeros(6, dtype=torch.float32), device=self.tensor_device)
        self.goal_threshold = 0.005

    def test_env(self):
        for i in range(1000):
            self.goal = self.gen_target()
            self.render_target()
            self.reset()
            self.scene.step()

    def gen_target(self):
        """生成有效目标点"""
        while True:
            goal = gs_rand_float(
                lower=torch.tensor([self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]], device=self.tensor_device),
                upper=torch.tensor([self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]], device=self.tensor_device),
                shape=(3,),
                device=self.tensor_device
            )
            dist_to_default = torch.linalg.norm(goal - self.default_ee_pos)
            if 0.4 < dist_to_default < 0.5 and goal[0] > 0.2 and goal[2] > 0.2:
                return goal

    def render_target(self):
        if self.visualize:
            self.scene.clear_debug_objects()
            self.scene.draw_debug_sphere(pos=self.goal.cpu().numpy(), radius=0.03, color=[0.1, 0.1, 0.9, 0.9])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[torch.Tensor, dict]:
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)

        self.robot.set_dofs_position(torch.zeros(6, device=self.tensor_device), self.motors_dof_idx)
        self.scene.step()
        self.default_ee_pos = self.robot.get_link("ee_center_body").get_pos()

        self.goal = self.gen_target()
        self.render_target()

        obs = self.get_observation()
        self.start_t = time.time()
        return obs.cpu().numpy(), {}
        
    def get_observation(self):
        joint_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        ee_pos = self.robot.get_link("ee_center_body").get_pos()
        return torch.cat((joint_pos, ee_pos), dim=0)

    def calc_reward(self, action, obs):
        dist_to_goal = torch.linalg.norm(obs[6:] - self.goal)
        
        # 非线性距离奖励
        if dist_to_goal < self.goal_threshold:
            distance_reward = 100.0
        elif dist_to_goal < 2*self.goal_threshold:
            distance_reward = 50.0
        elif dist_to_goal < 3*self.goal_threshold:
            distance_reward = 10.0
        else:
            distance_reward = 1.0 / (1.0 + dist_to_goal)
        
        # 动作相关惩罚
        action_diff = action - self.last_action
        smooth_penalty = 0.1 * torch.linalg.norm(action_diff)

        # 关节角度限制惩罚
        joint_penalty = 0.0
        for i in range(6):
            min_angle = self.jnt_range[i][0]
            max_angle = self.jnt_range[i][1]
            if obs[i] < min_angle:
                joint_penalty += 0.5 * (min_angle - obs[i])
            elif obs[i] > max_angle:
                joint_penalty += 0.5 * (obs[i] - max_angle)
        
        # 总奖励计算
        total_reward = distance_reward - smooth_penalty - joint_penalty
        # 更新上一步动作
        self.last_action = action.clone()
        
        return total_reward, dist_to_goal
    
    def step(self, action):
        # 将numpy动作转换为tensor
        action_tensor = torch.tensor(action, device=self.tensor_device, dtype=torch.float32)
        
        # 动作缩放
        scaled_action = torch.zeros(6, device=self.tensor_device, dtype=torch.float32)
        for i in range(6):
            scaled_action[i] = self.jnt_range[i][0] + (action_tensor[i] + 1) * 0.5 * (self.jnt_range[i][1] - self.jnt_range[i][0])
        
        # 执行动作
        self.robot.control_dofs_position(scaled_action, self.motors_dof_idx)
        self.scene.step()

        obs = self.get_observation()
        reward, dist_to_goal = self.calc_reward(action_tensor, obs)
        terminated = False
        if dist_to_goal < self.goal_threshold:
            terminated = True

        if not terminated:
            if time.time() - self.start_t > 20.0:
                reward -= 10.0
                print(f"[超时] 时间过长，奖励减半")
                terminated = True
        
        info = {
            'is_success': terminated and (dist_to_goal < self.goal_threshold),
            'distance_to_goal': dist_to_goal.item()
        }

        return obs.cpu().numpy(), reward.item(), terminated, False, info
    
    def seed(self, seed: Optional[int] = None) -> list[Optional[int]]:
        torch.manual_seed(seed)
        return [seed]
    
def train_ppo(
    n_envs: int = 24,
    total_timesteps: int = 40_000_000,
    model_save_path: str = "piper_ppo_reach_target",
    visualize: bool = False
) -> None:

    ENV_KWARGS = {'visualize': visualize}
    
    # 创建多进程向量环境
    env = make_vec_env(
        env_id=lambda: PiperEnv(**ENV_KWARGS),
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    
    # 策略网络配置
    POLICY_KWARGS = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )
    
    # PPO模型
    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        n_steps=2048,          
        batch_size=2048,       
        n_epochs=10,           
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard/piper_reach_target/"
    )
    
    print(f"并行环境数: {n_envs}, 总步数: {total_timesteps}")
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save(model_save_path)
    env.close()
    print(f"模型已保存至: {model_save_path}")


if __name__ == "__main__":
    # env = PiperEnv(True)
    # env.test_env()
    TRAIN_MODE = True
    MODEL_PATH = "/home/khalillee/genesis_workspace/piper_rl/piper_ppo_reach_target"
    if TRAIN_MODE:
        train_ppo(
            n_envs=2,                
            total_timesteps=400_000_000,
            model_save_path=MODEL_PATH,
            visualize = True
        )
    else:
        pass
