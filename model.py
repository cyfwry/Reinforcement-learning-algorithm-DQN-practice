import parl
from parl import layers
class PadModel(parl.Model):
    def __init__(self,act_dim):
        self.fc1=layers.fc(act_dim*20,act='relu')
        self.fc2=layers.fc(act_dim*20,act='relu')
        self.fc3=layers.fc(act_dim*20,act='relu')
        self.fc4=layers.fc(act_dim*20,act='relu')
        self.fc5=layers.fc(act_dim)
    
    def value(self,obs):
        out=self.fc1(obs)
        out=self.fc2(out)
        out=self.fc3(out)
        out=self.fc4(out)
        out=self.fc5(out)
        return out