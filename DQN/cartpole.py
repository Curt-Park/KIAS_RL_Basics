# -*- coding: utf-8 -*-

'''
Reference repos:
[1] https://github.com/keon/deep-q-learning
[2] https://gist.github.com/kkweon/5605f1dfd27eb9c0353de162247a7456

The overall implementation is a little modified from [2] with
some hyper parameter settings of [1].
'''

import numpy as np
import gym # OpenAI gym (a game simulator)
import random

from collections import deque
from keras.layers import (Input, Activation, Dense,
                          Flatten, RepeatVector, Reshape)
from keras.layers.convolutional import Conv2D
from keras.models import Model


class Agent:
    def __init__(self, env):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.create_model()

    def create_model(self, hidden_dims=[64, 64]):
        X = Input(shape=(self.input_dim, ))

        net = RepeatVector(self.input_dim)(X)
        net = Reshape([self.input_dim, self.input_dim, 1])(net)

        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)

        net = Flatten()(net)
        net = Dense(self.output_dim)(net)

        self.model = Model(inputs=X, outputs=net)
        self.model.compile('rmsprop', 'mse')

    def act(self, X, eps=1.0):
        if np.random.rand() < eps:
            return self.env.action_space.sample()

        X = X.reshape(-1, self.input_dim)
        Q = self.model.predict_on_batch(X)
        return np.argmax(Q, 1)[0]

    def train(self, X_batch, y_batch):
        return self.model.train_on_batch(X_batch, y_batch)

    def predict(self, X_batch):
        return self.model.predict_on_batch(X_batch)

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)


def create_batch(agent, memory, batch_size, gamma):
    sample = random.sample(memory, batch_size)
    sample = np.asarray(sample)

    s = sample[:, 0]
    a = sample[:, 1].astype(np.int8)
    r = sample[:, 2]
    s2 = sample[:, 3]
    d = sample[:, 4] * 1.

    X_batch = np.vstack(s)
    y_batch = agent.predict(X_batch)

    y_batch[np.arange(batch_size), a] = r + gamma * np.max(agent.predict(np.vstack(s2)), 1) * (1 - d)

    return X_batch, y_batch


def print_info(episode, score, eps):
    msg = f"[Episode {episode:>5}] Score: {score:>5} EPS: {eps:>3.2f}"
    print(msg)


def main():
    n_episode = 1000
    gamma = 0.95
    memory_size = 50000
    batch_size = 32
    eps = 1.0
    min_eps = 0.01
    eps_decay = 0.995
    replay_memory = deque()
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env._max_episode_steps = 500 # default is 200
    agent = Agent(env)
#    agent.load("./cartpole-dqn.h5")

    # CartPole-v0 Clear Condition
    for episode in range(n_episode):
        done = False
        score = 0
        state = env.reset()
        eps = max(min_eps, eps * eps_decay)

        while not done:
            env.render()
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            score += 1

            replay_memory.append([state, action, reward, next_state, done])

            if len(replay_memory) > memory_size:
                replay_memory.popleft()

            if len(replay_memory) > batch_size:
                X_batch, y_batch = create_batch(agent, replay_memory, batch_size, gamma)
                agent.train(X_batch, y_batch)

            state = next_state

        print_info(episode, score, eps)

#        if episode % 100 == 0:
#            agent.save("./save/cartpole-dqn.h5")

    env.close()


if __name__ == '__main__':
    main()
