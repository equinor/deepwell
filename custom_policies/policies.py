from stable_baselines.common.policies import FeedForwardPolicy

# Custom MLP policy of three non shared layers of size 128 each for the policy network
# and the value networkl
class ThreeOf128NonShared(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(ThreeOf128NonShared, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")


#Custom MLP policy of one shared layer of 55 followed by two non-shared for the value
#network and a single non-shared for the policy network

class OneShared55TwoValueOnePolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(OneShared55TwoValueOnePolicy, self).__init__(*args, **kwargs,
                                           net_arch=[55, dict(vf=[255,255],pf=[128])],
                                           feature_extraction="mlp")
