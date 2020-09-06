from keras.models import *
from keras.layers import *

from keras.optimizers import *

import random, numpy, math, Trend_Hearts_Env, scipy
from SumTree import SumTree
from keras import backend as K


HUBER_LOSS_DELTA = 2.0
LEARNING_RATE = 0.001

#-------------------- UTILITIES -----------------------
def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)



#-------------------- BRAIN ---------------------------


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        state_input = Input(batch_shape=(None, stateCnt))
        hidden_1 = Dense(264, activation='relu')(state_input)
        hidden_2 = Dense(132, activation='relu')(hidden_1)
        hidden_3 = Dense(264, activation='relu')(hidden_2)
        hidden_4 = Dense(132, activation='relu')(hidden_3)
        advantage = Dense(actionCnt, activation='linear')(hidden_4)
        value = Dense(1, activation='linear')(hidden_4)
        policy = Lambda(lambda x: x[0]- K.mean(x[0])+x[1] , output_shape=(actionCnt,))([advantage, value])
        model = Model(inputs=[state_input], outputs=[policy])
        #out_actions = Dense(actionCnt, activation='linear')(hidden_6)
        #model = Model(inputs=[state_input], outputs=[out_actions])
        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
        
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 200000

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.1

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            if int(float(s[2]))==1: #pass cards, need 3 actions
                a = np.array(random.sample(range(self.actionCnt), 3))
                return a
            else:
                a = random.randint(0, self.actionCnt-1)
                return a
        else:
            s = np.array([s])
            p = agent_brain.predict_p(s)[0]
            v = agent_brain.predict_v(s)[0]
            
            if int(float(s[0][2]))==1: #pass cards, need 3 actions
                try:
                    a = np.argpartition(self.brain.predictOne(s), -3)[-3:]
                    
                except: #only 1 item has nonzero probability, so just pick another random 2
                    print("ERROR")
            else:
                a = numpy.argmax(self.brain.predictOne(s))
                return a
    
    
    
    
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            #handle array
            if s_ is None:
                if t[a] isinstance(t[a], list):
                    aind = 0
                    for action in t[a]:
                        t[a][aind] = r
                        aind+=1
                else:
                    t[a] = r
            else:
                if t[a] isinstance(t[a], list):
                    aind = 0
                    for action in t[a]:
                        t[a][aind] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN
                        aind+=1
                else:
                    t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
    
        if int(float(s[2]))==1: #pass cards, need 3 actions
            a = np.array(random.sample(range(self.actionCnt), 3))
            return a
        else:
            a = random.randint(0, self.actionCnt-1)
            return a

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self):
        self.env = Trend_Hearts_Env()
        
    def run(self, agent):
        s = self.env.get_actionable_event()

        R = 0
        while True:         
            # self.env.render()
            a = agent.act(s)

            r = 0
            s_, r, done = self.env.step(a, s)

            # = np.clip(r, -1, 1)   # clip reward to [-1, 1]


            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) )
            agent.replay()            

            s = s_
            R += r

            if done:
                break
        #record final chips to evaluate performance
        fname = "ending_chips.data"
        f=open(fname, 'a+')
        f.write(str(R) + "\n")
        f.close()        
        print("Total reward:", R)

#-------------------- MAIN ----------------------------
env = Environment()

stateCnt  = env.env.get_observation_space_shape()
actionCnt = env.env.get_action_space_size()

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        env.run(randomAgent)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    while True:
        env.run(agent)
finally:
    agent.brain.model.save("Seaquest-DQN-PER.h5")