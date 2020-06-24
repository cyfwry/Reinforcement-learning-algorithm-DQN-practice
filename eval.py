from parl.algorithms import DQN
from pad import Paddle
from model import PadModel
from agent import PadAgent
from parl.utils import logger
from replay_memory import ReplayMemory
import numpy as np
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']=''

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.999  # discount factor of reward
  
def evaluate(agent,env):
    eval_reward=[]
    for _ in range(100):
        total_reward=0
        obs=env.reset()
        nums=0
        
        done=False
        while not done:
            action=agent.predict(obs)
            reward,next_obs,done=env.step(action)
            total_reward+=reward
            obs=next_obs
            nums+=1
            time.sleep(0.005)
       
        eval_reward.append(total_reward)
        
        
    return np.mean(eval_reward)

def eval():
    env=Paddle()
    action_dims=3
    obs_dims=5
    rpm=ReplayMemory(MEMORY_SIZE)
    model=PadModel(action_dims)
    algorithm=DQN(model,action_dims,GAMMA,LEARNING_RATE)
    agent=PadAgent(algorithm,obs_dim=obs_dims,act_dim=action_dims)
    
    # use this to test the model you want
    agent.restore('./dqn_model.ckpt')
    
    eval_reward=evaluate(agent,env)
    logger.info('test_reward:{}'.format(eval_reward))
        
if __name__=='__main__':
    eval()