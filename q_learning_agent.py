# Inspired by https://keon.io/deep-q-learning/
import numpy as np
from env import GameOfLifeEnv

class qLearning:
    '''
        In this algorithm, the learned action-value function, Q, directly approximates q ⇤ , the optimal
action-value function, independent of the policy being followed.
        Q-learning: Off-policy TD Control - Page 131
        Reinforcement Learning: An introduction
        Richard S. Sutton and Andrew G. Barto.
    '''

    def __init__(self, env, episodes, n_episodes=10000, gamma=0.99, epsilon=0.3,
                 alpha=0.9, render=False):
        self.episodes = episodes
        self.env = env
        self.render = render
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.dQ = {}  # Diccionario de Q

    # Main method qLearning
    def run(self):
        """Main loop that controls the execution of the agent"""
        scores = []
        for e in range(self.episodes):
            cumR = 0
            state = self.env.reset()
            done = False
            i = 0
            while not done:
                #
                if self.render:
                    self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)  # change action
                if done and reward == 0:
                    reward = 0

                self.learn(state, action, reward, next_state, done)
                i += 1
                cumR += reward
                state = next_state
            scores.append(cumR)
            if e % 100 == 0:
                print('[Episode {}] - Mean rewards {}.'.format(e, np.mean(scores)))

        return scores

    # Choose action optimized
    def choose_action(self, state):
        """Chooses the next action according to the model trained and the policy"""
        if np.random.sample() < self.epsilon:
            return self.env.action_space.sample()  # a random action is returned
        q = [self.dQ.get((state, a), 0) for a in range(self.env.action_space.n)]
        max_q = max(q)
        count = q.count(max_q)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(self.env.action_space.n) if q[i] == max_q]
            i = np.random.choice(best)
        else:
            i = q.index(max_q)
        action = i

        return action

    def learn(self, state, action, reward, next_state, done):
        # Previus state
        qOld = self.dQ.get((state, action), 0.0)
        if done:
            self.dQ[(
            state, action)] = self.alpha * reward
        else:
            new_max_q = max([self.dQ.get((next_state, a), 0.0) for a in
                             range(self.env.action_space.n)])
            self.dQ[(state, action)] = qOld + self.alpha * (
                        reward + self.gamma * new_max_q - qOld)

if __name__ == "__main__":
    qLearning(GameOfLifeEnv,1000,render=False)