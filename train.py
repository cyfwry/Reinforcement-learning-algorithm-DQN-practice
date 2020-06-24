from parl.algorithms import DQN
from pad import Paddle
from model import PadModel
from agent import PadAgent
from parl.utils import logger
from replay_memory import ReplayMemory
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES']=''

LEARN_FREQ = 5  # update parameters every 5 steps
MEMORY_SIZE = 20000  # replay memory size
MEMORY_WARMUP_SIZE = 200  # store some experiences in the replay memory in advance
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.999  # discount factor of reward

def run_eposide(agent,env,rpm):
    total_reward=0
    obs=env.reset()
    step=0
    while True:
        step+=1
        action=agent.sample(obs)
        reward,next_obs,done=env.step(action)
        rpm.append((obs,action,reward,next_obs,done))
        
        if(len(rpm)>MEMORY_WARMUP_SIZE) and (step%LEARN_FREQ):
            batch_obs,batch_action,batch_reward,batch_next_obs,batch_done=rpm.sample(BATCH_SIZE)
            train_loss=agent.learn(batch_obs,batch_action,batch_reward,batch_next_obs,batch_done)
            
        total_reward+=reward
        obs=next_obs
        if done:
            break
    #print(step)        
    return total_reward
    
def evaluate(agent,env):
    eval_reward=[]
    for _ in range(5):
        total_reward=0
        obs=env.reset()
        done=False

        while not done:            
            action=agent.predict(obs)
            reward,next_obs,done=env.step(action)
            total_reward+=reward
            obs=next_obs

        
        eval_reward.append(total_reward)
        
    return np.mean(eval_reward)

def main():
    env=Paddle()
    action_dims=3
    obs_dims=5
    rpm=ReplayMemory(MEMORY_SIZE)
    model=PadModel(action_dims)
    algorithm=DQN(model,action_dims,GAMMA,LEARNING_RATE)
    agent=PadAgent(algorithm,obs_dim=obs_dims,act_dim=action_dims)
    
    # use this to restore your model
    # agent.restore('./dqn_model.ckpt')
    while len(rpm)<MEMORY_WARMUP_SIZE:
        run_eposide(agent,env,rpm)
        
    max_eposide=1000
    
    for eposide in range(1,max_eposide+1):
        total_reward=run_eposide(agent,env,rpm)
        print(total_reward)
        if eposide%50==0:
            eval_reward=evaluate(agent,env)
            logger.info('eposide:{},test_reward:{}'.format(eposide,eval_reward))
        
            save_path = './dqn_model.ckpt'
            agent.save(save_path)
        
if __name__=='__main__':
    main()