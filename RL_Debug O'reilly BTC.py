# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 13:31:38 2021

@author: Pastor
"""

import pandas as pd
import numpy as np
import random

# Read the data
# set a dictionary with the variables and functions to create daily bars
ohlcv_dict = {'Open':'first', 'High':'max', 'Low':'min', 'Close': 'last', 'Volume':'sum'}

#path_file = "D:/Dropbox/Pastor/data/crypto-active_5min/BTC_5min.txt"
#btc = pd.read_csv(path_file, header = None, sep = ",", names = ["Date", "Open", "High", "Low", "Close", "Volume"])
path_file_es = "D:/Dropbox/Pastor/data/futures_unadjusted_5/ES_continuous_UNadjusted_5min.txt"
spy = pd.read_csv(path_file_es, header = None, sep = ",", names = ["Date", "Open", "High", "Low", "Close", "Volume"])
spy["Date"] = pd.to_datetime(spy['Date'])
spy.set_index('Date', inplace=True)
spy = spy.resample('1440Min').apply(ohlcv_dict).dropna()
returns_spy = spy.Close.pct_change().dropna()


path_file_nq = "D:/Dropbox/Pastor/data/futures_unadjusted_5/NQ_continuous_UNadjusted_5min.txt"
btc = pd.read_csv(path_file_nq, header = None, sep = ",", names = ["Date", "Open", "High", "Low", "Close", "Volume"])
btc["Date"] = pd.to_datetime(btc['Date'])             
btc.set_index('Date', inplace=True)
btc = btc.resample('1440Min').apply(ohlcv_dict).dropna()

# merge the two dataframes
stock_data = pd.merge(btc, returns_spy, right_index=True, left_index=True ).sort_index()
stock_data.columns = ["Open", "High", "Low", "Close", "Volume", "returns_spy"]

# get the restus of the two dataframes
returns = pd.DataFrame({
                'stocks': stock_data['Close'].rolling(window=2, center=False).apply(lambda x: x[1] / x[0] - 1),
                'spy': (stock_data['returns_spy']),
                            }, index=stock_data.index).dropna()


returns['risk_adjusted'] = returns.stocks - returns.spy # risk free returns

# we are actually using bollinger bands

# simple moving average of risk adjusted on the last 12 days
returns['risk_adjusted_moving'] = returns.risk_adjusted.rolling(window=12).apply(lambda x: x.mean())

# moving standard deviation of the last 12 days
returns['risk_adjusted_stedv'] = returns.risk_adjusted.rolling(window=12).apply(lambda x: x.std())

# upper band
returns['risk_adjusted_high']  = returns.risk_adjusted_moving + 1.5 * returns.risk_adjusted_stedv
#lower band
returns['risk_adjusted_low'] = returns.risk_adjusted_moving - 1.5 * returns.risk_adjusted_stedv

# if risk adjusted is above the upper band get 1
# if the risk adjusted is below the lower band get -1
returns['state'] = (returns.risk_adjusted > returns.risk_adjusted_high).astype('int') - \
                        (returns.risk_adjusted < returns.risk_adjusted_low).astype('int')

# training and testing set
midpoint = int(len(returns.index) / 2)
training_indexes = returns.index[:midpoint]
testing_indexes = returns.index[midpoint:]
        

factors = pd.DataFrame({'action': 0, 'reward': 0, 'state':0}, index = training_indexes)     

# value Iteration
# Q-Learning
# iterating on State, Action, Reward, and State prime
q = {0: {1:0, 0:0, -1:0}}

# Learning Q is expensive
# Dyna
T = np.zeros((3, 3, 3)) + 0.0001
R = np.zeros((3, 3))

def sharpe(holdings, returns):
    returns = holdings * (returns.stocks - returns.spy) # state time free risk return
    return np.nanmean(returns) / np.nanstd(returns) # average risk free returns by standard deviation


for i in range(1): # this loop iterates 100 times or until convergence is reach over the training data 
    last_row, last_date = None, None

    # check the rreturns on each row (depends on your dataframe can be day or hour or minute or every 30 min)
    for date, row in factors.iterrows():
        return_data = returns.loc[date] # get the current row
        
        if return_data.state not in q:
            q[return_data.state] = {1:0, 0:0, -1:0}

        if last_row is None or np.isnan(return_data.state):
            state = 0
            reward = 0
            action = 0
            
        else:
            state = int(return_data.state)

            # we add randomness to the state every 0.1% of the time
            if random.random() > 0.001:
                action = max(q[state], key=q[state].get)
            else:
                action = random.randint(-1, 1)
            
            # get the reward which is calculated as the action (buy = 1, sell = -1, don't do anything = 0)
            # if you outperform the spy you will get positive reward if you take action
            # we buy everytime the price is above the upper band of the bollinger band
            reward = last_row.action * (return_data.stocks - return_data.spy)

            factors.loc[date, 'reward'] = reward
            factors.loc[date, 'action'] = action
            factors.loc[date, 'state'] = return_data.state

            alpha = 0.01 # can be customize, so you can try 0.0001 if you want
            discount = 0.9

            update = alpha * (factors.loc[date, 'reward'] + discount * max(q[row.state].values()) - q[state][action])

            if not np.isnan(update):
                q[state][action] += update

            # Dyna
            action_idx = int(last_row.action + 1)
            state_idx = int(last_row.state + 1)
            new_state_idx = int(state + 1)

            T[state_idx][action_idx][state_idx] += 1

            R[state_idx][action_idx] = (1-alpha) * R[state_idx][action_idx] + alpha * reward
            
        last_date, last_row = date, factors.loc[date]
        
        
    for j in range(1):
            state_idx = random.randint(0, 2)
            action_idx = random.randint(0, 2)
            # choose a state which is sell, no action and buy, choose only one, with probability of T
            new_state = np.random.choice([-1, 0, 1], 1, p=T[state_idx][action_idx]/T[state_idx][action_idx].sum())[0]

            r = R[state_idx][action_idx]
            q[state][action] += alpha * (r + discount * max(q[new_state].values()) - q[state][action])

    sharpe = sharpe(factors.action, returns)

    if sharpe > 1.20:
        break

        print(f'For episode {i} we get an internal sharpe ratio of {self.sharpe(factors.action)}')
        
testing = pd.DataFrame({'action': 0, 'state': 0}, index = testing_indexes)
testing['state'] = returns.loc[testing_indexes, 'state']
testing['action'] = testing['state'].apply(lambda state: max(q[state], key = q[state].get))



def buy_and_hold(dates):
    return pd.Series(1, index = dates)

def buy_spy(dates):
    return pd.Series(0, index = dates)

def random(dates):
    return pd.Series(np.random.randint(-1, 2, size = len(dates)), index = dates)

def calc_returns(holdings):
    return pd.Series(returns.spy + holdings * returns.risk_adjusted, index=holdings.index)

def evaluate(holdings):
    return pd.Series(calc_returns(holdings) + 1).cumprod()

def q_holdings(training_indexes, testing_indexes):
    factors = pd.DataFrame({'action': 0, 'reward': 0, 'state':0}, index = training_indexes) 
                        
                        
                        
                        
                        
                        