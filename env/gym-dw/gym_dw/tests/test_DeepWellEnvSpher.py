import unittest
from envs import DeepWellEnvSpher
import numpy as np




class TestDeepWellEnvSpherV0(unittest.TestCase):

    def setUp(self):            #This is run before each test
        self.env = DeepWellEnvSpher()

    #def test_calc_angle_diff(self):
    #    self.assertEqual(DeepWellEnvSpher.calc_angle_diff(DeepWellEnvSpher,5.0, 5.0),1.0)

    def test_get_state(self):
        self.assertIsInstance(self.env.get_state(), np.ndarray)     #Check that get_state actually returns numpy array
    
    def test_step(self):
        p1 = np.array([self.env.x, self.env.y, self.env.z])
        
        state, reward, done, info = self.env.step(0)
        state, reward, done, info = self.env.step(1)
        state, reward, done, info = self.env.step(2)
        state, reward, done, info = self.env.step(3)
        state, reward, done, info = self.env.step(4)
        state, reward, done, info = self.env.step(5)
        state, reward, done, info = self.env.step(6)
        state, reward, done, info = self.env.step(7)
        state, reward, done, info = self.env.step(8)

        p2 = np.array([ info['x'], info['y'], info['z'] ])

        squared_dist = np.sum((p1-p2)**2, axis=0)
        dist = np.sqrt(squared_dist)
        self.assertTrue(89 <= dist <= 90)


if __name__ == '__main__':
    unittest.main()