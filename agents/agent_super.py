


class agent:

    def train(self, env, timesteps,modelpath, tensorboard_logs_path):
        raise NotImplementedError("Agent needs to implement train method")
        

    def load(self, modelpath, tensorboard_logs_path):
        raise NotImplementedError("Agent needs to implement load method")
        

    def retrain(self, env, timesteps, modelpath, tensorboard_logs_path):
        raise NotImplementedError("Agent needs to implement retrain method")


    def get_env_str(self,env):      #This returns the str name of an env, like 'DeepWellEnvSpher-v0' in gym.make('DeepWellEnvSpher-v0'). For use in make_vec_env()
        name = env.__str__()
        name = name.strip('<>')
        name = name.split('<')
        return name[1]


    