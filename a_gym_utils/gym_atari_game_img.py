import gym
import matplotlib.pyplot as plt

#ENV_NAME = "MsPacman-v0"
ENV_NAME = "SpaceInvaders-v0"

env = gym.make(ENV_NAME)
env.reset()
plt.imshow(env.render('rgb_array'))
plt.show()
