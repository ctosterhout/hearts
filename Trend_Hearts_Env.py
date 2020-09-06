import numpy as np 
#import hashlib
from websocket import create_connection
import json
import random


class Trend_Hearts_Env:

    card_value = {
            "2" : 0,
            "3" : 1,
            "4" : 2,
            "5" : 3,
            "6" : 4,
            "7" : 5,
            "8" : 6,
            "9" : 7,
            "T" : 8,
            "J" : 9,
            "Q" : 10,
            "K" : 11,
            "A" : 12
        }
    suit_value = {
            "S" : 0,
            "D" : 1,
            "H" : 2,
            "C" : 3
        }
        
        
    def __init__(self, player_name="test", player_number ="4", token="12345678", server_address = "ws://localhost:8080/", test=False):
        self.ws = ""
        self.player_name = player_name
        self.player_number=player_number
        self.server_address = server_address
        self.token = token
        self.reset_env()
        if not test:
            self.connect_to_server()
        else:
            print("Created player for test: " +str(player_name))

    def reset_env(self):
        self.card_status = np.zeros((52,))
        self.card_history_player = np.zeros((208,))
        self.score_cards = np.zeros((208,))
        self.passed_n_received = np.zeros((104,))
        self.phase = 0
        self.deal_num = 0
        self.round_num = 0
        self.reset = True
    
    def step(self,a, s):
        
        if isinstance(a, list):
            cards = []
            for action in a:
                cards.append(self.index_to_card(action))
        else:
            if a != "":
                cards = self.index_to_card(a)
            else:
                cards = ""
                a = 0
        if self.phase==1: 
            event_name ="pass_my_cards"
            data = {"dealNumber": int(float(s[0])),"cards": cards}
        elif self.phase==2:
            card_list = []
            card_list.append(cards)
            event_name ="expose_my_cards"
            data = {"dealNumber": int(float(s[0])),"cards": card_list}
        elif self.phase==3:
            event_name ="pick_card"
            data = {"dealNumber": self.deal_num, "roundNumber": self.round_num, "turnCard": cards}
        
        self.send_action(event_name, data)
    
        
        s_, event  = self.get_actionable_event()
        
        
        done = True if event == "game_end" else False
        r = self.calc_reward(s, s_, done)
        if done:
            self.reset_env()
            #self.ws.close()
            #self.connect_to_server()
        return s_, r, a, done 
        
        
        
    def render(self):
        pass
        
    def get_actionable_event(self):
        data = {}
        event= ""
        try:
            result = self.ws.recv()
            msg = json.loads(result)
            event = msg["eventName"]
            data = msg["data"]
        except Exception as e:
            pass
            #print("Connection error " + str(e))
            #self.connect_to_server()
        while event not in  ["your_turn", "pass_cards", "expose_cards", "game_end"]:
            try:
                result = self.ws.recv()
                msg = json.loads(result)
                event = msg["eventName"]
                data = msg["data"]
                if event == 'new_deal':
                    self.reset_env()
                #print("EVENT " + str(event))
                #state = self.data_to_state(data, event)
                
            except Exception as e:
                pass
                #print("Connection error "  + str(e))
                #self.connect_to_server()
        state = self.data_to_state(data, event)
        return state, event
        
    
        
        
    def calc_reward(self, s, s_, done):

        reward = 0
        if done:
            rlist =[]
            rlist.append(s_[36])
            rlist.append(s_[37])
            rlist.append(s_[38])
            rlist.append(s_[39])
            if s_[35+int(self.player_number)] == max(rlist):
                reward += 1
            else: #s_[35+int(self.player_number)] == min(rlist):
                reward += -1

        #elif s_[16] == 1:
        #    r2list =[]
        #    r2list.append(s_[32] + s_[36])
        #    r2list.append(s_[33] + s_[37])
        #    r2list.append(s_[34] + s_[38])
        #    r2list.append(s_[35] + s_[39])
        #    r3list =[]
        #    r3list.append(s_[32])
        #    r3list.append(s_[33])
        #    r3list.append(s_[34])
        #    r3list.append(s_[35])
        #    if s_[31+int(self.player_number)] +s_[35+int(self.player_number)]  == max(r2list):
        #        reward +=1
        #    if s_[31+int(self.player_number)] == max(r3list):
        #        reward += 1
        #    else:
        #        reward += -1
     
        #print(str(self.player_name) +" Reward: " + str(reward) + " Round: " +str(s[1]) + " " + str(s_[1]) )
        return reward
        
        
    def send_action(self, event_name, data):
        #print(str(self.player_name) + " Sending: " + str(event_name) + ", " + str(data))
        message = {"eventName": event_name,"data": {}}
        message["data"] = data
        self.ws.send(json.dumps(message))

        
    def get_observation_space_shape(self):
        return 720
        
    def get_action_space_size(self):
        return 52
        
    def connect_to_server(self):
        self.ws = create_connection(self.server_address)
        self.ws.send(json.dumps({"eventName": "join","data": {"playerNumber": self.player_number, "playerName": self.player_name, "token":self.token}}))
        #self.player_md5 = hashlib.md5(self.player_name.encode('utf-8')).hexdigest()
        print("Connecting " + str(self.player_name) + " " + str(self.server_address) + " " + str(self.player_number))

    
################################ Stuff to convert data from server #############################
    def data_to_state(self, data, event):
    

        #print("-----------starting to create state--------------")
        deal                                = self.get_data_deal(data) #0-3
        round                               = self.get_data_round(data) #4-16
        phase                               = self.get_data_phase(data,event) #17-19 -1: pass cards, 2: expose ah, 3: pick cards, 0: ukknown
        ah_exposed                          = self.get_data_ah_exposed(data) #20-23 is the ace of hearts exposed and by the player number (should change to 1-hot)
        received_from                       = self.get_data_received_from(data)#24-27 who we received cards from
        passed_to                           = self.get_data_passed_to(data)#28-31 - who we passed cards to
        d_scores, g_scores                  = self.get_data_player_scores(data) #32-39  -d_score is deal score, g_score is game score
        card_history, card_history_player   = self.get_data_card_history(data, event) #40-91 - card_history is the current cards being played. card_history_player is which player won the played the card
        card_status                         = self.get_data_card_status(data, event) #92-143 - cards I have, array of 52 that is multi-hot encoded
        player_num                          = self.get_data_player_num() #144-147 - our player number
        passed_n_received                   = self.get_data_passed_received(data,event) #148-251 
        valid_cards                         = self.get_data_valid_cards(data) #252-303
        #card_history_player  304-511
        score_cards                         = self.get_data_score_cards(data) #512-719

        state = np.concatenate((deal,round,phase,ah_exposed,received_from,passed_to,d_scores,g_scores,
                        card_history, card_status,player_num,passed_n_received,valid_cards, card_history_player,
                        score_cards),axis=None)


        #np.set_printoptions(suppress=True)
        #print ("----------done with creating state------------")
        state = state.astype(int)
        return state
        
        
    def get_data_player_num(self):
        player_num = np.zeros((4,))
        player_num[self.player_number-1] = 1
        return player_num
        
    def get_data_deal(self, data):
        deal_num_one_hot = np.zeros((4,))
        if "dealNumber" in data:
            self.deal_num = data["dealNumber"]
            deal_num_one_hot[data["dealNumber"]-1] = 1
        return deal_num_one_hot
    
    def get_data_round(self, data):
        round_num_one_hot = np.zeros((13,))
        if "roundNumber" in data:
            self.round_num = data["roundNumber"]
            round_num_one_hot[data["roundNumber"]-1] = 1
        return round_num_one_hot
    
    def get_data_phase(self, data, event):
        phase_num = np.zeros((3,))
        if event == 'pass_cards': #'game_prepare', 'new_game','new_deal', 'round_end'
            phase_num[0] = 1
            self.phase = 1
        elif event == 'expose_cards': #'recieve_opponent_cards'
            phase_num[1] = 1
            self.phase = 2
        elif event in ['your_turn', 'new_round', 'pass_cards_end']: #'turn_end'
            phase_num[2] = 1
            self.phase = 3
        return phase_num
       
    
    
    def get_data_ah_exposed(self, data):
        ah_exposed = np.zeros((4,))
        if "players" in data:
            for player in data['players']:
                if "exposedCards" in player and player['exposedCards']:
                    ah_exposed[player['playerNumber']-1] = 1
        return ah_exposed
    
    def get_data_received_from(self, data):
        recieved_from = np.zeros((4,))
        if 'self' in data:
            r_p_name = data['self']['receivedFrom']
            for player in data['players']:
                if player['playerName'] == r_p_name:
                    recieved_from[player['playerNumber']-1] = 1
        return recieved_from
    
    def get_data_passed_to(self, data):
        passed_to = np.zeros((4,))
        if "players" in data:
            for player in data['players']:
                if "receivedFrom" in player and player['receivedFrom'] == self.player_name:
                    passed_to[player['playerNumber']-1] = 1
        return passed_to
        
    def get_data_player_scores(self, data):
        p_deal_scores = []
        p_game_scores = []
        if "players" in data:
            for player in data['players']:
                if "dealScore" in player:
                    p_deal_scores.append(player['dealScore'])
                else:
                     p_deal_scores.append(0)
                if "gameScore" in player:
                    p_game_scores.append(player['gameScore'])
                else:
                    p_game_scores.append(0)
            return p_deal_scores, p_game_scores
        return [0,0,0,0],[0,0,0,0]
    
    
    def get_data_card_history(self, data, event):
        card_history = np.zeros((52,))
        if 'players'in data:
            for player in data['players']:
                if "roundCard" in player:
                    if player['roundCard']:
                        card_index = self.card_to_index(player['roundCard'])
                        card_history[card_index] = 1
                        #create card_history_player
                        self.card_history_player[int(player['playerNumber'] - 1)*52 + card_index] = 1
        player_history = self.card_history_player
        return card_history, player_history
            
    
    
    def get_data_card_status(self, data, event):
        my_cards = []
        if "self" in data:
            self.card_status = np.zeros((52,))
            for card in data['self']['cards']:
                self.card_status[self.card_to_index(card)] = 1
        status = self.card_status
        return status

        
    def get_data_passed_received(self,data,event):
        if event == 'pass_cards_end':
            for player in data['players']:
                if player['playerNumber'] == self.player_number:
                    self.passed_n_received = np.zeros((104,))
                    picked_cards = []
                    received_cards = []
                    for card in player['pickedCards']:
                        self.passed_n_received[self.card_to_index(card)] = 1
                    for card in player['receivedCards']:
                        self.passed_n_received[self.card_to_index(card) +52] = 1
        pnr = self.passed_n_received
        return pnr
        
    def get_data_valid_cards(self, data):
        if self.phase == 1:
            valid_actions = np.ones((52,))
        elif self.phase == 2:
                valid_actions = np.zeros((52,))
                valid_actions[38] = 1
        elif self.phase == 3:
            valid_actions = np.zeros((52,))
            if 'self' in data:
                for card in data['self']['candidateCards']:
                    valid_actions[int(self.card_to_index(card))] = 1
        return valid_actions
        
    def get_data_score_cards(self, data):
        if 'players'in data:
            for player in data['players']:
                if "scoreCards" in player:
                    if player['scoreCards']:
                        for card in player['scoreCards']:
                            card_index = self.card_to_index(card)
                            self.score_cards[int(player['playerNumber'] - 1)*52 + card_index] = 1
        sc = self.score_cards
        return sc
################################ Utility functions for getters #############################
    def convert_card_format(self, card):
        if len(card) != 2:
            return
        return card[0] + card[1]
        
        

############################## Convert between card string, action, and index ################
    
    def card_to_index(self, c):
        return self.card_value[self.convert_card_format(c)[0]] + self.suit_value[self.convert_card_format(c)[1]]*13

        
        
    def index_to_card(self, i):
        if isinstance(i, list):
            i2 = i[0]
        else:
            i2=i
        suit_num, card_num= divmod(i2, 13)
        return (list(self.card_value.keys())[list(self.card_value.values()).index(card_num)]) + (list(self.suit_value.keys())[list(self.suit_value.values()).index(suit_num)])

    def get_card_index(self, c):
        c_index = -1
        card_id = self.card_value[self.convert_card_format(c)[0]] + self.suit_value[self.convert_card_format(c)[1]]*13
        if self.card_status[card_id] >-1 and self.card_status[card_id] < 13:
            c_index = card_id
        return c_index
        