import parl
from parl import layers
import paddle.fluid as fluid
import numpy as np

class PadAgent(parl.Agent):
    def __init__(self,algorithm,obs_dim,act_dim,global_step=0,update_target_steps=200,e_greed=0.01,e_greed_decrement=1e-6):
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        
        self.global_step=global_step
        self.update_target_steps=update_target_steps
        
        self.e_greed=e_greed
        self.e_greed_decrement=e_greed_decrement

        super().__init__(algorithm)
        
    def build_program(self):
        self.learn_program=fluid.Program()
        self.pred_program=fluid.Program()
        with fluid.program_guard(self.learn_program):
            obs=layers.data(name='obs',shape=[self.obs_dim],dtype='float32')
            action=layers.data(name='action',shape=[1],dtype='int32')
            reward=layers.data(name='reward',shape=[],dtype='float32')
            next_obs=layers.data(name='next_obs',shape=[self.obs_dim],dtype='float32')
            terminal=layers.data(name='terminal',shape=[],dtype='bool')
            self.cost=self.alg.learn(obs,action,reward,next_obs,terminal)
        
        with fluid.program_guard(self.pred_program):
            obs=layers.data(name='obs',shape=[self.obs_dim],dtype='float32')
            self.value=self.alg.predict(obs)
            
    def predict(self,obs):
        obs=np.expand_dims(obs,axis=0)
        obs=obs.astype('float32')
        feed={'obs':obs}
        pred_Q=self.fluid_executor.run(self.pred_program,feed=feed,fetch_list=[self.value])[0]
        pred_Q=np.squeeze(pred_Q,axis=0)
        act=np.argmax(pred_Q)
        return act
        
    def learn(self,obs,action,reward,next_obs,terminal):
        if self.global_step%self.update_target_steps==0:
            self.alg.sync_target()
        
        self.global_step+=1
        
        action=np.expand_dims(action,axis=-1)
        feed={'obs':obs.astype('float32'),'action':action.astype('int32'),'reward':reward,'next_obs':next_obs.astype('float32'),'terminal':terminal}
        loss=self.fluid_executor.run(self.learn_program,feed=feed,fetch_list=[self.cost])
        return loss
        
    def sample(self,obs):
        prob=np.random.rand()
        if prob<self.e_greed:
            act=np.random.choice(self.act_dim,1)
        else:
            act=self.predict(obs)
            
        self.e_greed=max(0.01,self.e_greed-self.e_greed_decrement)
        return act
            