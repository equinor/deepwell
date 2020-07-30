
from gym.envs.registration import register
register(
    id='DeepWellEnv-v0',
    entry_point='gym_dw.envs:DeepWellEnv'
)
register(
    id='DeepWellEnv-v2',
    entry_point='gym_dw.envs:DeepWellEnvV2'
)

register(
    id='DeepWellEnv3d-v0',
    entry_point='gym_dw.envs:DeepWellEnv3D'
)

register(
    id='DeepWellEnvSpher-v0',
    entry_point='gym_dw.envs:DeepWellEnvSpher'
)

register(
    id='DeepWellEnvSpherSmallObs-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherSmallObs'
)

# Register different levels of DeepWellEnvSpher for level training
register(
    id='DeepWellEnvSpherlevel1-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level1'
)

register(
    id='DeepWellEnvSpherlevel2-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level2'
)

register(
    id='DeepWellEnvSpherlevel3-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level3'
)

register(
    id='DeepWellEnvSpherlevel4-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level4'
)

register(
    id='DeepWellEnvSpherlevel5-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level5'
)

register(
    id='DeepWellEnvSpherlevel6-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level6'
)

register(
    id='DeepWellEnvSpherlevel7-v0',
    entry_point='gym_dw.envs.DeepWellEnvSpherLevels:Level7'
)

