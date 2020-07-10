
from gym.envs.registration import register
register(
    id='DeepWellEnv-v0',
    entry_point='gym_dw.envs:DeepWellEnv'
)

register(
    id='DeepWellEnv2-v0',
    entry_point='gym_dw.envs:DeepWellEnv2'
)

