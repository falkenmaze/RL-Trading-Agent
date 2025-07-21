import gym
import pandas as pd 
import yfinance as yf 
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from TradingEnv import TradingEnv

# FETCH DATA 
ticker = "SPY"
df = yf.download(ticker, start="2020-01-01", end="2023-12-31")
print(df.columns)
# CREATE AND WRAP ENVIRONMENT
# env = DummyVecEnv([lambda: TradingEnv(df, render_mode="human")])
env = TradingEnv(df, render_mode='human')

# INITIALIZE DQN MODEL
# model = DQN(
# 	policy = "MlpPolicy",
# 	env = env,
# 	learning_rate = 1e-3,
# 	buffer_size = 10000,
# 	learning_starts = 1000,
# 	batch_size = 32,
# 	gamma = 0.99,
# 	train_freq = 1,
# 	target_update_interval = 100,
# 	verbose = 1
# 	)

# # TRAIN THE MODEL
# model.learn(total_timesteps = 50000)
# # SAVE THE MODEL
# model.save("dqn_training_model")

# TEST THE MODEL
obs = env.reset()
model = DQN.load("dqn_training_model", env=env)
done = False
step = 0
while not done:
	action, _ = model.predict(obs)
	obs,reward,done,_ = env.step(action)
	if step % 100 == 0:
		env.render()
	step += 1