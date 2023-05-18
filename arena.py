import gymnasium as gym
import numpy as np

class Arena:
    def __init__(self):
        self.cheetah = gym.make('HalfCheetah-v4', render_mode='rgb_array')
    
    def fight(self, behaviour, max_steps = 1000, render=False, repetitions=3) -> int:
        '''
        :param cheetah: A func that takes a array with 17 elements (observation) and return a numpy array with 6 elements (action)
        :return: A integer that represents the score of the cheetah
        '''
        
        if render:
            self.cheetah = gym.make('HalfCheetah-v4', render_mode='rgb_array', ctrl_cost_weight=0.0)
            self.cheetah = gym.wrappers.RecordVideo(self.cheetah, "export/" + str(render) + ".mp4")
        else:
            self.cheetah = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.0)
        
        try:
            total_reward = 0
            for _ in range(repetitions):
                # observation  = self.cheetah.reset_model()
                observation, info = self.cheetah.reset()
                terminated, truncated = False, False
                steps        = 0
                
    
                while not (terminated or truncated) and steps < max_steps:# not (terminated or truncated) and
                    action = behaviour(observation)
                    action = action[0]
                    action = np.clip(action, -1.0, 1.0)
                    observation, reward, terminated, truncated, info = self.cheetah.step(action)
                    steps        += 1
                    total_reward += reward
                    if render:
                        self.cheetah.render()
                    
            if render:
                self.cheetah.close()

        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
        finally:
            return total_reward/repetitions