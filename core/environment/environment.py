import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from math import copysign
from core.tools import safe_div
pd.options.mode.chained_assignment = None


INITIAL_ACCOUNT_BALANCE = 100_000


class StockTradingEnv(gym.Env):
    """ A stock trading environment for OpenAI gym
        StockTradingEnv uses the data cluster for pre-processing
        This is only used to stage data and in the supervised learning stage
    """

    metadata = {'render.modes': ['human']}
    ACTION_SPACE_SIZE = 3 # Buy, Sell or Hold


    def __init__(
        self,
        collection,
        look_back_window,
        max_steps=300):

        super(StockTradingEnv, self).__init__()

        # Constants
        self.LOOK_BACK_WINDOW = look_back_window
        self.max_steps = max_steps

        # Set number range as df index and save date index
        self.collection = collection

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)



    def _next_observation(self):
        # Define the span for the observation
        span = (
            self.current_step - (self.LOOK_BACK_WINDOW-1), 
            self.current_step
        )
        
        # Request datapack to process the data in span
        df_st, df_lt = self.dp.data_process(span)

        return {'st':df_st, 'lt':df_lt}




    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "close"])
        action_type = action

        self.current_price = self.df.loc[self.current_step, "close"]
        self.buy_n_hold = (self.current_price / self.initial_price) - 1
        comission = 0 # The comission is applied to both buy and sell
        amount = 0.3

        if action_type == 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / self.current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * self.current_price * (1 + comission)

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought
            self.shares_bought_total += additional_cost
            self.buy_triggers += 1

        elif action_type == 2:
            # Sell amount % of shares held
            self.shares_sold = int(self.shares_held * amount)
            self.balance += self.shares_sold * self.current_price * (1 - comission)
            self.shares_held -= self.shares_sold
            self.total_shares_sold += self.shares_sold
            self.total_sales_value += self.shares_sold * self.current_price
            self.sell_triggers += 1

        # Save amount
        self.amounts.append(amount)

        # Update the net worth
        self.net_worth = self.balance + self.shares_held * self.current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0



    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_date = self.date_index[self.current_step]

        # Take next step
        self.current_step += 1
        done = False
        
        # Check if there are no more steps in the data or we have met the maximum amount of steps
        if self.current_step == self.start_step+self.max_steps or self.current_step == len(self.df.loc[:, 'open'].values)-1:
            done = True

        # Update next observation
        obs = self._next_observation()

        # Done if net worth is negative
        if not done:
            done = self.net_worth <= 0

        reward = (self.net_worth / INITIAL_ACCOUNT_BALANCE) - 1

        return obs, reward, done


    def _gen_initial_step(self):
        
        # To generate initial step
        self.current_step = random.randint(
            self.LOOK_BACK_WINDOW + 50, len(self.df.loc[:, 'open'].values)-self.max_steps
            ) #add 50 due to that datapack needs to calc the features and remove nan rows



    def reset(self):

        # Sample a data pack from the cluster and setup df
        self.dp = np.random.choice(self.collection, replace=True)
        self.df = self.dp.df
        self.date_index = self.dp.date_index

        # Reset the state of the environment to an initial state
        self.ticker = self.dp.ticker
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.shares_bought_total = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.buy_triggers = 0
        self.sell_triggers = 0
        self.amounts = list()
        

        # Set initial values
        self._gen_initial_step()
        self.start_step = self.current_step + 1
        self.initial_price = self.df.loc[self.start_step, "close"]
        self.initial_date = self.date_index[self.start_step]
        self.buy_n_hold = 0


        return self._next_observation()



    def render(self):
        ''' Render the environment to the screen '''

        _stats = {
            'ticker': self.ticker,
            'amountBalance': round(self.balance),
            'amountAsset': round(self.total_sales_value),
            'netWorth': self.net_worth,
            'netWorthChng': round( self.net_worth / INITIAL_ACCOUNT_BALANCE , 3),
            'profit': round(self.net_worth - INITIAL_ACCOUNT_BALANCE),
            'buyAndHold': round(  self.df.close[self.current_step] / self.initial_price , 3),
            'fromToDays': (self.date_index[self.current_step] - self.initial_date).days
            }

        for statName, stat in _stats.items():
            print(f'{statName}: {stat}')
        return





if __name__ == '__main__':
    pass