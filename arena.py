import gymnasium as gym

class Arena:
    def __init__(self):
        self.cheetah = gym.make('HalfCheetah-v4' ) # , ctrl_cost_weight=0.1, ....)
        # pass
    
    def fight(self, behaviour, max_steps = 1000) -> int:
        '''
        :param cheetah: A func that takes a array with 17 elements (observation) and return a numpy array with 6 elements (action)
        :return: A integer that represents the score of the cheetah
        '''
        # TODO: Implement simulation of the cheetah in the enviroment HalfCheetah

        # Spravis .. get_ ( "Cheetah-v4" )
        #   To ti vrati Cheetah Environment.
        #       Environment ma:
        #           step(action) - Sem davas vystup neuronky. Vracia ti to Observation (vstup pre NN),  reward, _ , _ , info {obsahuje: x_position, x_velocity, reward_run, reward_ctrl}
        #           _get_obs() to ti vrati nejaku position a velocity ale asi nemas preco pouzivat
        #           reset_model() - resetne to model, vratit ti to uvodnu observation zasumenu na vstupe ofc.
        #           
        try:
            total_reward = 0
            observation  = self.cheetah.reset_model()
            # frame_delay  = 1.0 / fps # actually need seconds per frame for sleep method
            done         = False
            steps        = 0
            while not done and steps < max_steps:
                # if render:
                    #environment.render() # wrappers.Monitor already calls this
                    #time.sleep(frame_delay) # wrappers.Monitor already have fps monitoring
                action = behaviour(observation)
                observation, reward, done, _, info = self.cheetah.step(action)
                steps        += 1
                total_reward += reward
            # if render:
            #     print("Total reward: ", str(total_reward))
            #     print("Total steps: ", str(steps))
        finally:
            # if render:
            #     time.sleep(1)
            #     environment.reset_video_recorder()
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