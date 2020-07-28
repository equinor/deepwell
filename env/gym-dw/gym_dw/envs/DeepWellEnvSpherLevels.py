from gym_dw.envs.DeepWellEnvSpher import DeepWellEnvSpher


class Level1(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 0
        self.maxdeltaZ_target = 0
        self.numhazards = 0
        self.min_radius = 100
        self.max_radius = 100
        self.reset()
    

class Level2(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 0
        self.maxdeltaZ_target = 100
        self.numhazards = 0
        self.min_radius = 100
        self.max_radius = 100
        self.reset()


class Level3(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 100
        self.maxdeltaZ_target = 100
        self.numhazards = 0
        self.min_radius = 100
        self.max_radius = 100
        self.reset()


class Level4(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 200
        self.maxdeltaZ_target = 500
        self.numhazards = 0
        self.min_radius = 100
        self.max_radius = 100
        self.reset()


class Level5(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 200
        self.maxdeltaZ_target = 500
        self.numhazards = 2
        self.min_radius = 100
        self.max_radius = 100
        self.reset()


class Level6(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 200
        self.maxdeltaZ_target = 500
        self.numhazards = 7
        self.min_radius = 100
        self.max_radius = 100
        self.reset()


class Level7(DeepWellEnvSpher):
    
    def __init__(self):
        DeepWellEnvSpher.__init__(self)
        self.deltaY_target = 200
        self.maxdeltaZ_target = 500
        self.numhazards = 10
        self.min_radius_hazard = 75
        self.max_radius_hazard = 150
        self.min_radius = 75
        self.max_radius = 150
        self.reset()



if __name__ == '__main__' :
    env = DeepWellEnvSpher()
    env.reset()
    for _ in range(10):
        action = 0 #env.action_space.sample()
        print(env.actions_dict[action])
        print("Step: ", _ , " this is what the current state is:")
        print(env.step(action))