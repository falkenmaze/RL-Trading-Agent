import gym
import numpy as np 
from gym import spaces 
import pandas as pd 

def compute_rsi(series, period=14):
		delta = series.diff() #calculating difference in prices (if diff > 0, gain else loss)
		gain = delta.clip(lower=0)
		loss = -delta.clip(upper=0)
		avg_gain = gain.rolling(window=period).mean() #calculating average gain over period
		avg_loss = loss.rolling(window=period).mean() #calculating average loss over period
		rs = avg_gain/(avg_loss + 1e-10) #prevent division by zero
		rsi = 100 - (100 / (1 + rs))
		return rsi


class TradingEnv(gym.Env):
	metadata = {"render_modes": ["human"], "render_fps": 4}
	def __init__(self, df, initial_cash=10000, window_size=10, render_mode=None):
		super(TradingEnv, self).__init__()

		self.df = df.reset_index()
		self.render_mode = render_mode
		self.current_step = 0
		self.initial_cash = initial_cash
		self.window_size = window_size
		self.cash = self.initial_cash
		self.shares_held = 0 
		self.total_value = self.initial_cash

		# Precompute Indicators
		self.df['MA10'] = self.df['Close'].rolling(window=10).mean()
		self.df['MA20'] = self.df['Close'].rolling(window=20).mean()
		# If MA10 crosses above MA20, bullish; If MA10 crosses below MA20, bearish
		self.df['RSI'] = compute_rsi(self.df['Close']) # RSI > 70 (overbought); RSI < 30 (oversold)

		self.df.dropna(inplace=True)
		self.df.reset_index(drop=True, inplace=True)
		# Allow 0-5 shares to be bought or sold in one step
		# Actions: 0 = hold; 1-5 = buy 1-5; 6-10 = sell 1-5
		self.action_space = spaces.Discrete(11)

		# Observations: [Price, history + indicators, no. of shares held]
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,5),  dtype=np.float32)
		self.reset()

	def reset(self):
		self.current_step = self.window_size
		self.cash = self.initial_cash
		self.shares_held = 0
		self.net_worth = self.initial_cash
		return self._get_observation()

	def _get_observation(self):
		frame = self.df.iloc[self.current_step - self.window_size: self.current_step]
		frame = frame.copy()
		frame['Close'] = frame['Close'] / frame['Close'].iloc[0] - 1
		frame['MA10'] = frame['MA10'] / frame['MA10'].iloc[0] - 1
		frame['MA20'] = frame['MA20'] / frame['MA20'].iloc[0] - 1
		frame['RSI'] = frame['RSI'] / 100.0
		obs = frame[['Close', 'MA10', 'MA20', 'RSI']].values
		shares = np.full((self.window_size, 1), self.shares_held)
		obs = np.hstack((obs, shares))
		return obs.astype(np.float32)

	def _buy(self, amount):
		available_cash = self.cash
		price = self.df.iloc[self.current_step][('Close', 'SPY')]
		total_cost = amount * price

		if total_cost < available_cash:
			self.shares_held += amount
			self.cash -= total_cost

	def _sell(self, amount):
		if self.shares_held >= amount:
			price = self.df.iloc[self.current_step][('Close', 'SPY')]
			self.shares_held -= amount
			self.cash += amount * price

	def step(self, action):
		current_price = self.df.iloc[self.current_step][('Close', 'SPY')]
		if action == 0:
			pass # Hold
			reward = -0.0001
		elif 1 <= action <= 5: # Buy
			self._buy(action)
			prev_net_worth = self.net_worth
			self.net_worth = self.cash + self.shares_held * current_price
			reward = (self.net_worth - prev_net_worth) / prev_net_worth

		elif 6 <= action <= 10: # Sell
			self._sell(action - 5)
			prev_net_worth = self.net_worth
			self.net_worth = self.cash + self.shares_held * current_price
			reward = (self.net_worth - prev_net_worth) / prev_net_worth


		self.current_step += 1
		done = bool(self.current_step >= len(self.df) - 1 or self.net_worth <= 0)
		obs = self._get_observation()
		return obs,reward,done,{}

	def render(self, mode="human"):
		if self.render_mode == 'human':
			print(f'Step: {self.current_step}')
			print(f'Price: {self.df.iloc[self.current_step][("Close", "SPY")]:.2f}')
			print(f'Cash: {self.cash:.2f}')
			print(f'Shares Held: {self.shares_held}')
			print(f'Net Worth: {self.net_worth:.2f}')
			print("\n\n")


