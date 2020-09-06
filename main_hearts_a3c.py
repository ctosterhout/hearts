import numpy as np
import tensorflow as tf


from io import StringIO
import time, random, threading, os, Trend_Hearts_Env

from keras.models import *
from keras.layers import *
from keras import backend as K

import sys

#-- constants

NONTRAIN = False  #set to true when competing. It will just create 1 thread with min eps

RUN_TIME = 600000
if not NONTRAIN:
    THREADS = 32 #example, specifying 2 will create 3 threads, 0 for test, 1 and 2 for training
    OPTIMIZERS = 5
else:
    THREADS = 0
    OPTIMIZERS = 1
THREAD_DELAY = 0.0001
GAMMA = 0.99

N_STEP_RETURN = 15
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.99
EPS_STOP  = 0.05
EPS_STEPS = 100000

MIN_BATCH = 32
if not NONTRAIN:
    LEARNING_RATE = .0015
else:
    LEARNING_RATE = .00001
LOSS_V = .5 # v loss coefficient ~.5
LOSS_ENTROPY = .01  # entropy coefficient ~.01



###################################




########################################################################
summary_writer = tf.summary.FileWriter('logs')
game_counter = 0
name_i = 0
bot_counter = 0
#server_counter1 = 6
server_counter1 = 8
server_counter2 = 1
class Environment(threading.Thread):
    stop_signal = False
    

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS, test=False, pname="craig", token="1234", pnumber=1, saddress="ws://localhost:8081"):
        global bot_counter
        global server_counter2, server_counter1
        global server_address
        global summary_writer
    
        threading.Thread.__init__(self)
        
        self.render = render
        if test == False:
            player_name = "craig" + str(bot_counter) 
            server_address = "ws://localhost:80" + str(server_counter1) + str(server_counter2)
            player_number = bot_counter
            player_token = player_number
        else:
            player_name = pname 
            server_address = saddress
            player_number = pnumber
            player_token=token
        bot_counter +=1
        if bot_counter > 4:
            bot_counter = 1
            server_counter2 +=1
        if server_counter2 > 9:
            server_counter2 = 0
            server_counter1 +=1
        #server_counter2 +=1
        time.sleep(.1)

        self.env = Trend_Hearts_Env.Trend_Hearts_Env(player_name = player_name, player_number = player_number, server_address=server_address, token = player_token, test=test)
        self.agent = Agent_Body(eps_start, eps_end, eps_steps)

    def run(self):
        while not self.stop_signal:
            self.play_game()

    def play_game(self):
        global game_counter
        global steps
        R = 0
        s,_ = self.env.get_actionable_event()
        while True:
            time.sleep(THREAD_DELAY)
            
            #if self.render: self.env.render()
            a = self.agent.act(s)
            steps +=1
            s_, r, a, done = self.env.step(a, s)


            
            if done:
                s_ = None
                
            np.set_printoptions(suppress=True)
            #print("remembering- Action: " + str(a) +", Reward: "+str(r) + ", Done: "+ str(done))
            self.agent.train(s, a, r, s_)
            
            s = s_
            R += r
            
            if done or self.stop_signal:
                break

        game_counter+=1
        summary=tf.Summary()
        summary.value.add(tag='Reward', simple_value = R)
        summary_writer.add_summary(summary, steps)
        print("Server: " + self.env.server_address + ", Player: " + str(self.env.player_number) + ", Game:" + str(game_counter) + ", Final step:" + str(steps) + ", Total reward:" + str(R) )

    def stop(self):
        self.stop_signal = True
        

        


########################################################################
steps = 0
class Agent_Body:
    
    def __init__(self, eps_start, eps_end, eps_steps):
        
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        
        
        self.memory = []
        self.R = 0
        

    def act(self, s):
        global steps
        eps = self.get_epsilon()
        
        cards = s[92:144]
        valid_cards= s[252:304]
        if random.random() < eps:
            if int(float(s[17]))==1: #pass cards, need 3 actions
                a = []
                for it in range(0,3):
                    a.append(random.choice([i for i, x in enumerate(cards) if x == 1]))
                    cards[a][it] = 0
            elif int(float(s[18]))==1: #expose it if we got it
                randnum = random.randint(1,101)
                if cards[38] == 1 and randnum > 49:
                    a= 38
                else:
                    a=""
            else:
                a = random.choice([i for i, x in enumerate(valid_cards) if x == 1])
            return a
        else:
            s = np.array([s])
            p = agent_brain.predict_p(s)[0]
            
            invalid_indexes = np.where(valid_cards==0)[0]
            p[invalid_indexes] = 0

                
            #v = agent_brain.predict_v(s)[0]
            if int(float(s[0][17]))==1: #pass cards, need 3 actions
                a = []
                a = sorted(range(len(p)), key=lambda i: p[i])[-3:]
            else:
                a = sorted(range(len(p)), key=lambda i: p[i])[-1:]
            return a
            
    
    def get_epsilon(self):
        global steps
        if steps >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + steps * (self.eps_end - self.eps_start) / self.eps_steps
    
    def train(self, s, a, r, s_):
        global steps
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _ ,_ ,_, s_ = memory[n-1]
            
            return s, a, self.R, s_
            
        a_onehot = np.zeros(NUM_ACTIONS)
        if isinstance(a, list): #list of three actions
            for it in range(0,len(a)):
                a_onehot[a[it]] = 1
        else:
            a_onehot[a] = 1
        
        self.memory.append((s, a_onehot, r, s_))

        self.R = (self.R + r *GAMMA_N) / GAMMA
        
        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                agent_brain.train_push(s, a, r, s_)
                
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            
            self.R = 0
        
        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            agent_brain.train_push(s, a, r, s_)
            
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)
            


########################################################################

class Agent_Brain:
    global summary_writer
    train_queue = [ [], [], [], [], [] ]
    lock_queue = threading.Lock()
    
    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        if not NONTRAIN:
            K.set_learning_phase(0) #set to 1 when testing to remove dropouts
        else:
            K.set_learning_phase(1)
        
        self.model = self._build_model()
        self.graph = self._build_graph(self.model)
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        
        summary_writer.add_graph(self.default_graph)
        self.merge = tf.summary.merge_all()
        
        if os.path.isfile("my_model_weights.h5"):
            self.model.load_weights("my_model_weights.h5")
            print("loading previous session weights")
        self.saver = tf.train.Saver()
        if os.path.isfile("model.ckpt.index"):
            print("loading previous session data")
            self.saver.restore(self.session, "model.ckpt")
        self.default_graph.finalize()
            
        
    def _build_model(self):
        state_input = Input(batch_shape=(None, NUM_STATE))
        hidden_1 = Dense(1024, activation='relu')(state_input)
        hidden_2 = Dense(512, activation='relu')(hidden_1)
        hidden_3 = Dense(1024, activation='relu')(hidden_2)
        hidden_4 = Dense(512, activation='relu')(hidden_3)
        
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(hidden_4)
        out_value = Dense(1, activation='linear')(hidden_4)
        
        model = Model(inputs=[state_input], outputs=[out_actions, out_value])
        model._make_predict_function()
        
        return model
        
        
        
        
    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))
        

        
        p, v = model(s_t)
        
        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v
        
        loss_policy = log_prob * tf.stop_gradient(advantage)
        loss_value = LOSS_V * tf.square(advantage)
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        tf.summary.scalar("loss_total", loss_total)

        
        
        #optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        #minimize = optimizer.minimize(loss_total)


        gradients, variables = zip(*optimizer.compute_gradients(loss_total))
        #gradients, _ = tf.clip_by_global_norm(gradients,5.0)
        minimize = optimizer.apply_gradients(zip(gradients, variables))

        #for index, grad in enumerate(gradients):
        #    tf.summary.histogram(gradients[index].name, gradients[index])
        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        for gradient, variable in zip(gradients, variables):
            tf.summary.histogram("gradients/" + variable.name + " l2_norm", l2_norm(gradient))
            tf.summary.histogram("variables/" + variable.name + " l2_norm", l2_norm(variable))
            tf.summary.histogram("gradients/" + variable.name, gradient)
            tf.summary.histogram("variables/" + variable.name, variable)

        return s_t, a_t, r_t, minimize
        
    def optimize(self):
        global steps
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)
            return
            
        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:
                return
                
            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [ [], [], [], [], [] ]
            
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)
        
        if len(s) > 5*MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))
        
        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask

        
        
        s_t, a_t, r_t, minimize = self.graph
        
        
        
        summary, _ = self.session.run([self.merge, minimize], feed_dict={s_t: s, a_t: a, r_t: r})
        summary_writer.add_summary(summary, steps)
        
    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)
            
            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
                
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)
                
    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v
            
    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p
            
    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v
               


########################################################################

class Optimizer(threading.Thread):
    stop_signal = False
    
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        while not self.stop_signal:
            agent_brain.optimize()
            
    def stop(self):
        self.stop_signal = True
        
########################################################################

argv_count=len(sys.argv)
if argv_count>2:
    player_name = sys.argv[1]
    player_number = sys.argv[2]
    token= sys.argv[3]
    connect_url = sys.argv[4]
else:
    player_name="craig"
    player_number=1
    token=123
    connect_url= "ws://localhost:8081"



env_test = Environment(render=False, eps_start=0., eps_end=0.,test=True, pname=player_name, pnumber=player_number, token=token, saddress=connect_url)
NUM_STATE = env_test.env.get_observation_space_shape()
NUM_ACTIONS = env_test.env.get_action_space_size()
NONE_STATE = np.zeros(NUM_STATE)


agent_brain = Agent_Brain()


try:
    if NONTRAIN == False:
        envs = [Environment() for i in range(THREADS)]
        opts = [Optimizer() for i in range(OPTIMIZERS)]
        for o in opts:
            o.start()
        
        for e in envs:
            e.start()
            
        time.sleep(RUN_TIME)
        
        for e in envs:
            e.stop()
        for e in envs:
            e.join()
            
        for o in opts:
            o.stop()
        for o in opts:
            o.join()
finally:
    if NONTRAIN == True:
        env_test.env.connect_to_server()
        env_test.run()
    agent_brain.model.save_weights('my_model_weights.h5')
    #agent_brain.model.save('my_full_model_weights.h5')
    #agent_brain.saver.save(agent_brain.session, "\\model.ckpt")