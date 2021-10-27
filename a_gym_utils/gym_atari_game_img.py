import gym
import matplotlib.pyplot as plt

#ENV_NAME = "MsPacman-v0"
# ENV_NAME = "SpaceInvaders-v0"
ENV_NAME = "PongNoFrameskip-v4"

env = gym.make(ENV_NAME)
env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
    env, grayscale_obs=True )
env = gym.wrappers.FrameStack(env, num_stack=4, lz4_compress=True)

obs = env.reset()

print(obs.shape)
print(env.action_space)
print(env.action_space.n)
print(env.get_action_meanings())

plt.imshow(env.render('rgb_array'))
plt.show()
