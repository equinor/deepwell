
from gym.envs.registration import register
register(
    id='DeepWellEnv-v0',
    entry_point='gym_dw.envs:DeepWellEnv'
)

register(
    id='DwDiffeqEnv-v0',
    entry_point='gym_dw.envs:DwDiffeqEnv'
)

