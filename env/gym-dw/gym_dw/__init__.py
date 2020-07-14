
from gym.envs.registration import register
register(
    id='DeepWellEnv-v0',
    entry_point='gym_dw.envs:DeepWellEnv'
)
register(
    id='DeepWellEnv-v2',
    entry_point='gym_dw.envs:DeepWellEnv_v2'
)

register(
    id='DeepWellEnv3d-v0',
    entry_point='gym_dw.envs:DeepWellEnv3D'
)

