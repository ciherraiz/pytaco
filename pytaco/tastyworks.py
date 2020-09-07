from pathlib import Path
import numpy as np
import pandas as pd

STRATEGIES ={
    'Credit Put Spread' : {'pattern': [('BUY', 'PUT'), ('SELL', 'PUT')],
                           'direction': 'UP', 
                           'weekly_max_positions': 2,
                           'monthly_max_positions': 2},
    'Credit Call Spread' : {'pattern': [('SELL', 'CALL'), ('BUY', 'CALL')],
                           'direction': 'DOWN',
                           'weekly_max_positions': 1,
                           'monthly_max_positions': 1},
    'Short Iron Condor' : {'pattern': [('BUY', 'PUT'), ('SELL', 'PUT'), ('SELL', 'CALL'), ('BUY', 'CALL')],
                           'direction': 'SIDEWAYS',
                           'weekly_max_positions': 0,
                           'monthly_max_positions': 0},
    'Short Put': {'pattern': [('SELL', 'PUT')],
                            'direction': 'UP',
                            'weekly_max_positions': 3,
                            'monthly_max_positions': 0},
    'Short Call': {'pattern': [('SELL', 'CALL')],
                            'direction': 'DOWN',
                            'weekly_max_positions': 0,
                            'monthly_max_positions': 0},
    'Long Put': {'pattern': [('BUY', 'PUT')],
                            'direction': 'DOWN',
                            'weekly_max_positions': 0,
                            'monthly_max_positions': 0},
    'Long Call': {'pattern': [('BUY', 'CALL')],
                            'direction': 'UP',
                            'weekly_max_positions': 0,
                            'monthly_max_positions': 0}
}


class TWAccount:
    def __init__(self, path):
        self.data = pd.read_csv(path, thousands=',')
        # convert to datetime datatype
        self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
        self.data['Expiration Date'] = pd.to_datetime(self.data['Expiration Date'])

    def _to_trades(self):
        # select trade rows
        trades = self.data[(self.data['Type']=='Trade') | (self.data['Type']=='Receive Deliver')].sort_values(by=['Date', 'Strike Price'])
        
        # Trade type
        trades.loc[trades['Type'] == 'Trade', 'Type'] = 'TRADE'
        trades.loc[trades['Description'].str.contains('Forward split'), 'Type'] = 'FORWARD SPLIT'
        trades.loc[trades['Description'].str.contains('expiration'), 'Type'] = 'EXPIRATION'
        
        # Expiration
        trades.loc[trades['Type']=='EXPIRATION', 'Action'] = 'EXP'
        trades.loc[trades['Type']=='EXPIRATION', 'Buy or Sell'] = 'EXP'
        trades.loc[trades['Type']=='EXPIRATION', 'Open or Close'] = 'CLOSE'

        # Other type trades
        trades.loc[((trades['Action']=='BUY_TO_OPEN') | (trades['Action']=='BUY_TO_CLOSE')), 'Buy or Sell'] = 'BUY'
        trades.loc[((trades['Action']=='SELL_TO_OPEN') | (trades['Action']=='SELL_TO_CLOSE')), 'Buy or Sell'] = 'SELL'
        trades.loc[((trades['Action']=='BUY_TO_OPEN') | (trades['Action']=='SELL_TO_OPEN')), 'Open or Close'] = 'OPEN'
        trades.loc[((trades['Action']=='BUY_TO_CLOSE') | (trades['Action']=='SELL_TO_CLOSE')), 'Open or Close'] = 'CLOSE'

        # Setting the same time for all trades of a split
        splits = trades[trades['Type']=='FORWARD SPLIT'].groupby(by=['Underlying Symbol', 'Expiration Date']).first().reset_index()
        trades = trades.merge(splits, how='left', on=['Underlying Symbol', 'Expiration Date'], suffixes=('','_split'))
        trades.loc[trades['Type']=='FORWARD SPLIT', 'Date'] = trades['Date_split']

        trades.fillna(value=0, inplace=True)

        # Sum all trades of same order
        trades = trades.groupby([
                            'Type',
                            'Action',
                            'Symbol',
                            'Instrument Type',
                            'Underlying Symbol',
                            'Expiration Date',
                            'Strike Price',
                            'Call or Put',
                            'Buy or Sell',
                            'Open or Close',
                            'Multiplier'])\
                            .agg(Date=('Date', np.max),
                                Value=('Value', np.sum),
                                Quantity=('Quantity', np.sum),
                                Commissions=('Commissions', np.sum),
                                Fees=('Fees', np.sum))\
                            .reset_index()\
                            .sort_values(by=['Date', 'Open or Close', 'Strike Price'])
        
        self.trades = trades


