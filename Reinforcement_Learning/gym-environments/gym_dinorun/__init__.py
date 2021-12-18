from gym.envs.registration import register

register(id='DinoRun-v0',
         entry_point='gym_dinorun.envs:DinoRunEnv',
         )
