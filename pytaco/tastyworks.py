from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd

WEEKLY_THRESHOLD = 15
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


Action = namedtuple('Action', ['date', 
                            'underlying', 
                            'buy_sell', 
                            'call_put', 
                            'open_close', 
                            'strike', 
                            'quantity',
                            'value',
                            'commissions', 
                            'fees', 
                            'multiplier', 
                            'expiration', 
                            'strategy_id'])

Leg = namedtuple('Action', ['buy_sell', 
                            'call_put',
                            'open_close',
                            'strike',
                            'quantity',
                            'value',
                            'multiplier',
                            'expiration'])

class TWAccount:
    def __init__(self, path):
        self.data = pd.read_csv(path, thousands=',')
        # convert to datetime datatype
        self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True)
        self.data['Expiration Date'] = pd.to_datetime(self.data['Expiration Date'])

        self._to_trades()
        self._link_by_strategy()
        self._strategy_calculations()


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

    def _add_action(self, row, positions, strategies, id):
        st = strategies.copy()
        pos = positions.copy()

        action = Action(date=row['Date'],
                        underlying=row['Underlying Symbol'],
                        buy_sell=row['Buy or Sell'],
                        call_put=row['Call or Put'],
                        open_close=row['Open or Close'],
                        strike=row['Strike Price'],
                        quantity=row['Quantity'],
                        value=row['Value'],
                        commissions=row['Commissions'],
                        fees=row['Fees'],
                        multiplier=row['Multiplier'],
                        expiration=row['Expiration Date'],
                        strategy_id=id)
    
        # if this position wasn't created before
        if not row['Symbol'] in pos:
            pos[row['Symbol']] = []

        if not id in st:
            st[id] = []

        pos[row['Symbol']].append(action)
        st[id].append(action)
        return st, pos

    def _link_by_strategy(self):
        current_date = None
        current_id = None
        strategy_actions = {}
        position_actions = {}
        self.trades['strategy_id'] = np.nan
        last_id = 0

        for _, row in self.trades.iterrows():
            if row['Open or Close']=='OPEN': 
                # The first action in this trade is open a position -> new strategy
                # It's important to sort the trades by CLOSE-OPEN order
                if current_date != row['Date']:
                    current_date = row['Date']
                    last_id += 1
                    current_id = last_id

                strategy_actions, position_actions = self._add_action(row,
                                                                position_actions,
                                                                strategy_actions,
                                                                current_id)
            else: 
                # It's a new action of same trade
                # If the first action is a CLOSE, it's a adjustment probably
                if current_date != row['Date']:
                    current_date = row['Date']
                    for a in reversed(position_actions[row['Symbol']]):
                        # Search the strategy for this close action
                        if row['Quantity'] == a.quantity:
                            # We assume that a trade only contains one strategy
                            current_id = a.strategy_id
                            break
                            
                strategy_actions, position_actions = self._add_action(row, 
                                                                    position_actions, 
                                                                    strategy_actions, 
                                                                    current_id)

        actions = [a for a in strategy_actions.values()]
        actions = [y for x in actions for y in x ] #flattened
        self.trades = pd.DataFrame.from_records(actions, columns=Action._fields).sort_values(by=['strategy_id', 'date', 'open_close', 'strike'])


    def _strategy_pattern(self, legs, dte):
        pattern = []
        for l in legs:
            pattern.append((l[0], l[1]))

        for k, v in STRATEGIES.items():
            if v['pattern'] == pattern:
                return k
        
        print(f"ERROR: strategy not recognized {legs}")
        return None

    def _strategy_measures(self, strategy_name, legs):
        max_profit = None
        max_loss = None
        limit = None
        multiplier = legs[0].multiplier

        max_profit = sum([l.value for l in legs])

        if strategy_name == 'Credit Put Spread' or strategy_name == 'Credit Call Spread':
            max_loss = abs(legs[1].strike - legs[0].strike) * legs[0].quantity * multiplier
        elif strategy_name == 'Short Iron Condor': 
            max_loss = (legs[1].strike - legs[0].strike) * legs[0].quantity * multiplier
        elif strategy_name == 'Short Put':
            max_loss = legs[0].strike * legs[0].quantity * multiplier
        elif strategy_name == 'Short Call':
            max_loss = np.nan
        elif strategy_name == 'Long Put' or strategy_name == 'Long Call':
            max_loss = abs(legs[0].value)
            max_profit = np.nan

        strikes = [l.strike for l in legs]
        value = sum([l.value for l in legs])

        if STRATEGIES[strategy_name]['direction'] == 'UP':
            limit = max(strikes)
            if strategy_name == 'Long Call':
                limit += abs(value)/multiplier
            
        elif STRATEGIES[strategy_name]['direction'] == 'DOWN':
            limit = min(strikes)
            if strategy_name == 'Long Put':
                limit -= abs(value)/multiplier

        # SIDEWAYS¿?¿?

        return max_loss, max_profit, limit

    def _strategy_calculations(self):
        
        for _, strategy_trades in self.trades.groupby(['strategy_id']):
            open_legs = []
            strategy_closed = False
            strategy_created = False
            dte = None

            for _, row in strategy_trades.iterrows():
                if (row['open_close']=='OPEN'): 
                    if strategy_created == False:
                        leg = Leg(buy_sell=row['buy_sell'],
                                call_put=row['call_put'],
                                open_close=row['open_close'],
                                strike=row['strike'],
                                quantity=row['quantity'],
                                value=row['value'],
                                multiplier=row['multiplier'],
                                expiration=row['expiration'])
                        
                        open_legs.append(leg)
                        # days to expiration
                        dte = (row['expiration'].date() - row['date'].date()).days + 1
                        strategy_type = 'weekly' if dte < WEEKLY_THRESHOLD else 'monthly'
                        strategy_closed = False
                    else:
                        # reopen the trade state by a adjustment o roll
                        strategy_closed = False
                else:
                    strategy_created = True
                    strategy_closed = True

            name = self._strategy_pattern(open_legs, dte)
            max_loss, max_profit, limit = self._strategy_measures(name, open_legs)
            self.trades.loc[strategy_trades.index.values, 'strategy_name'] = name
            self.trades.loc[strategy_trades.index.values, 'max_loss'] = max_loss
            self.trades.loc[strategy_trades.index.values, 'max_profit'] = max_profit
            self.trades.loc[strategy_trades.index.values, 'dte'] = dte
            self.trades.loc[strategy_trades.index.values, 'strategy_type'] = strategy_type
            self.trades.loc[strategy_trades.index.values, 'strategy_state'] = 'CLOSED' if strategy_closed else 'OPEN'
            self.trades.loc[strategy_trades.index.values, 'direction'] = STRATEGIES[name]['direction']
            self.trades.loc[strategy_trades.index.values, 'limit'] = limit
