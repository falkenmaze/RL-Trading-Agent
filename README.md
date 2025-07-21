# Deep Reinforcement Learning SPY Trading Agent

This project implements a Deep Q-Network (DQN) agent trained to trade the SPY ETF using historical price data. It uses a custom OpenAI Gym environment to simulate a realistic trading experience with capital constraints, daily market steps, indicators and dynamic portfolio tracking.

## 🚀 Project Overview

The agent learns to **buy, sell, or hold SPY** stock to maximize net worth over time using deep reinforcement learning.

✅ **Current Results**:
- Initial capital: $10,000  
- Final net worth (after 1000+ steps): **~$16,500**  
- Learned strategy: buy low, hold during uptrends, and take profits at peaks

> This is an early version; future versions will improve reward shaping and model generalization.

## Reinforcement Learning Approach

- **Algorithm**: Deep Q-Learning (DQN)
- **Observation Space**:
  - Current price of SPY
  - Moving Average computed over 10 days
  - Moving Average computed over 20 Days
  - RSI commputed over 14 days
  - Number of shares held
- **Action Space**:
  - 0 → Hold
  - 1-5 → Buy 1-5 shares
  - 6-10 → Sell 1-5 shares
- **Reward Function**:
  - +ve reward on profitable trades
  - Small penalty for holding to encourage active trading
  - Encourages maximizing portfolio value over time

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/spy-trading-dqn.git
cd spy-trading-dqn
```
### 2. Install the Requirements
```bash
pip install -r requirements.txt
```
### 3. Use the trained model or Train your own Model
```bash
python train_dqn.py
```

