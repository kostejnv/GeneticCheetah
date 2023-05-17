import gymnasium as gym
import numpy as np

class Arena:
    def __init__(self):
        self.cheetah = gym.make('HalfCheetah-v4' ) # , ctrl_cost_weight=0.1, ....)
    
    def fight(self, behaviour, max_steps = 1000) -> int:
        '''
        :param cheetah: A func that takes a array with 17 elements (observation) and return a numpy array with 6 elements (action)
        :return: A integer that represents the score of the cheetah
        '''     
        try:
            
            # observation  = self.cheetah.reset_model()
            observation, info = self.cheetah.reset()
            terminated, truncated = False, False
            steps        = 0
            total_reward = 0

            while not (terminated or truncated) and steps < max_steps:# not (terminated or truncated) and 
                action = behaviour(observation)
                action = action[0]
                action = np.clip(action, -1.0, 1.0)
                observation, reward, terminated, truncated, info = self.cheetah.step(action)
                steps        += 1
                total_reward += reward

        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
        finally:
            return total_reward
        
        
# import time
# from gym import wrappers

# def simulate(actor, environment, max_steps=1000, video_postfix='Best', render=False, fps=30):
#     """Executes a full episode of actor in environment, for at most max_steps time steps
# If render is True, will render simulation on screen at ~fps frames per second."""
#     if render:
#         directory = "cs-169-rendered/gen" + str(video_postfix)
#         environment = wrappers.Monitor(environment, directory, force=True)
#     try:
#         total_reward = 0
#         observation  = environment.reset()
#         # frame_delay  = 1.0 / fps # actually need seconds per frame for sleep method
#         done         = False
#         steps        = 0
#         while not done and steps < max_steps:
#             # if render:
#                 #environment.render() # wrappers.Monitor already calls this
#                 #time.sleep(frame_delay) # wrappers.Monitor already have fps monitoring
#             action = actor.react_to(observation)
#             observation, reward, done, info = environment.step(action)
#             steps        += 1
#             total_reward += reward
#         if render:
#             print("Total reward: ", str(total_reward))
#             print("Total steps: ", str(steps))
#     finally:
#         if render:
#             time.sleep(1)
#             environment.reset_video_recorder()

#     return total_reward