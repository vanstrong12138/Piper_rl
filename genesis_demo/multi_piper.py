import genesis as gs
import torch

########################## 初始化 ##########################
gs.init(backend=gs.cpu)

########################## 创建场景 ##########################
scene = gs.Scene(
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3.5, -1.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
    ),
    rigid_options = gs.options.RigidOptions(
        dt                = 0.01,
    ),
)

########################## 实体 ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/agilex_piper/piper.xml'),
)

########################## 构建 ##########################

# 创建10个并行环境
B = 10
scene.build(n_envs=B, env_spacing=(1.0, 1.0))

# 控制所有机器人
franka.control_dofs_position(
    torch.tile(
        torch.tensor([0, 0, 0, -1.0, 0, 0, 0, 0], device=gs.device), (B, 1)
    ),
)

# 控制第1、5、7个环境的机器人
# franka.control_dofs_position(
#     position = torch.zeros(3, 8, device=gs.device),
#     envs_idx = torch.tensor([1, 5, 7], device=gs.device),
# )

for i in range(1000):
    scene.step()