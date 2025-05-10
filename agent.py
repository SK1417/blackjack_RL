import random 
from tqdm import tqdm
    
class Player:

    def __init__(self, dealers_card, game_instance: 'Game'):
        self.cards = [game_instance.deal_card(), game_instance.deal_card()]
        self.dealers_card = dealers_card
        self.player_turn = True
        self.bust = False

    def get_ace_count(self):
        return self.cards.count('A')

    def holds_usable_ace(self):
        return 'A' in self.cards and self.request_sum() <= 21
    
    def take_action(self, action: str, game_instance: 'Game'):
        if action == 'hit':
            self.cards += [game_instance.deal_card()]
            if self.request_sum() > 21:
                self.player_turn = False
                self.bust = True
        elif action == 'stick':
            self.player_turn = False

    def get_dealer_card(self):
        if self.dealers_card == 'A':
            return 11
        elif self.dealers_card in ['J', 'Q', 'K']:
            return 10
        else:
            return self.dealers_card

    def get_state(self):
        return self.request_sum(), self.get_dealer_card(), self.holds_usable_ace(), self.cards.count('A')
    
    def request_sum(self):
        sum = 0
        ace_count = 0
        for i in self.cards:
            if i == 'A':
                sum += 11
                ace_count += 1
            elif i in ['J', 'Q', 'K']:
                sum += 10
            else:
                sum += i
        
        while sum > 21 and ace_count > 0:
            sum -= 10
            ace_count -= 1

        return sum

class Dealer:

    def __init__(self, dealer_first_card, game_instance: 'Game'):
        self.cards = [dealer_first_card, game_instance.deal_card()]
        self.showing_card = dealer_first_card

    def request_sum(self):
        sum = 0
        ace_count = 0
        for i in self.cards:
            if i == 'A':
                sum += 11
                ace_count += 1
            elif i in ['J', 'Q', 'K']:
                sum += 10
            else:
                sum += i
        
        while sum > 21 and ace_count > 0:
            sum -= 10
            ace_count -= 1
        
        return sum
    
    def take_turn(self, game_instance: 'Game'):
        while self.request_sum() <= 17:
            self.cards += [game_instance.deal_card()]

    
class Game:
    
    def __init__(self):
        self.cards = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
        self.face_card_value = 10
        self.card_space = self.cards * 4
        random.shuffle(self.card_space)
        self.player = None
        self.dealer = None
    
    def deal_card(self):
        if not self.card_space:
            self.card_space = self.cards*4
            random.shuffle(self.card_space)
        return self.card_space.pop()
    
    def reset(self):
        self.card_space = self.cards * 4
        random.shuffle(self.card_space)
        dealer_first_card = self.deal_card()
        self.player = Player(dealer_first_card, self)
        self.dealer = Dealer(dealer_first_card, self)
        return self.player.get_state()
    
    def step(self, action):
        self.player.take_action(action, self)
        if self.player.bust:
            return self.player.get_state(), -1, True
        if not self.player.player_turn:
            self.dealer.take_turn(self)
            return self.calculate_reward()
        return self.player.get_state(), 0, False
    
    def calculate_reward(self):
        player_sum = self.player.request_sum()
        dealer_sum = self.dealer.request_sum()

        if self.player.bust:
            return (self.player.get_state(), -1, True)
        elif dealer_sum > 21:
            return (self.player.get_state(), 1, True)
        elif player_sum > dealer_sum:
            return (self.player.get_state(), 1, True)
        elif player_sum == dealer_sum:
            return (self.player.get_state(), 0, True)
        elif player_sum < dealer_sum:
            return (self.player.get_state(), -1, True)
        

class RLAgent:

    def __init__(self):
        self.policyMap = {}
        self.actionValuePairs = {}
        self.returns = {}
        self.game = Game()
        self.counts = {}

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(['hit', 'stick'])
        else:
            if state not in self.policyMap.keys():
                return random.choice(['hit', 'stick'])
            else:
                return self.policyMap[state]

    def reset(self):
        self.game.reset()

    def learn(self, num_episodes, gamma, initial_eps, eps_decay, min_eps):

        epsilon = initial_eps

        print('[LOG] learning......')
        for episode in tqdm(range(num_episodes)):
            state = self.game.reset()
            start_action = random.choice(['hit', 'stick'])
            done = False
            
            episode_history = []
            action = start_action
            next_state, reward, done = self.game.step(action)
            episode_history.append((state, start_action, reward))
            state = next_state

            while not done:
                action = self.choose_action(state, epsilon)
                next_state, reward, done = self.game.step(action)
                episode_history.append((state, action, reward))
                state = next_state
            
            visited_pairs = set()
            episode_return = 0
            for i in range(len(episode_history)-1, -1, -1):
                state, action, reward = episode_history[i]
                episode_return += gamma*reward
                if (state, action) not in visited_pairs:
                    self.counts[(state, action)] = self.counts.get((state, action), 0) + 1
                    alpha = 1.0/self.counts[(state, action)]
                    current_q = self.actionValuePairs.get((state, action), 0)
                    self.actionValuePairs[(state, action)] = current_q + alpha*(episode_return - current_q)
                    visited_pairs.add((state, action))
            
            ### policy improvement
            for state, _ in visited_pairs:
                best_action = None
                max_value = -float('inf')

                for action in ('hit', 'stick'):
                    if (state, action) in self.actionValuePairs:
                        if self.actionValuePairs[(state, action)] > max_value:
                            best_action = action
                            max_value = self.actionValuePairs[(state, action)]
                
                if best_action:
                    self.policyMap[state] = best_action
            
            epsilon = max(min_eps, epsilon*eps_decay)

def main():
    agent = RLAgent()
    num_episodes = 5000000
    initial_eps = 0.5
    eps_decay = 0.99999
    min_eps = 0.001
    gamma = 0.999

    agent.learn(num_episodes, gamma, initial_eps, eps_decay, min_eps)

    total_reward = 0
    num_eval_eps = 10000
    print('[LOG] evaluating....')
    for _ in tqdm(range(num_eval_eps)):
        state = agent.reset()
        done = False
        while not done:
            action = agent.choose_action(state, 0)
            next_state, reward, done = agent.game.step(action)
            total_reward += reward
            state = next_state
    
    print(f'[LOG] average reward over {num_eval_eps} is {total_reward/num_eval_eps:.4f}')

    

if __name__ == '__main__':
    main()