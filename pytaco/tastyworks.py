from collections import namedtuple
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

WEEKLY_THRESHOLD = 15
STRATEGIES ={
    'Put Credit Spread' : {'pattern': [('BUY', 'PUT'), ('SELL', 'PUT')],
                           'direction': 'UP', 
                           'weekly_max_positions': 0,
                           'monthly_max_positions': 0},
    'Call Credit Spread' : {'pattern': [('SELL', 'CALL'), ('BUY', 'CALL')],
                           'direction': 'DOWN',
                           'weekly_max_positions': 0,
                           'monthly_max_positions': 0},
    'Short Iron Condor' : {'pattern': [('BUY', 'PUT'), ('SELL', 'PUT'), ('SELL', 'CALL'), ('BUY', 'CALL')],
                           'direction': 'SIDEWAYS',
                           'weekly_max_positions': 0,
                           'monthly_max_positions': 0},
    'Short Put': {'pattern': [('SELL', 'PUT')],
                            'direction': 'UP',
                            'weekly_max_positions': 0,
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

        self.data['Commissions'].replace('--', 0, inplace=True)
        self.data['Commissions'] = self.data['Commissions'].astype(float)

        self.account_size = self.data[self.data['Description']=='Wire Funds Received']['Value'].sum().squeeze()

        self.trades = self._to_trades()
        self.trades = self._link_by_strategy()
        self.strategies = self._strategy_calculations()


    def _to_trades(self):
        # select trade rows
        trades = self.data[(self.data['Type']=='Trade') | ((self.data['Type']=='Receive Deliver') & (self.data['Instrument Type']=='Equity Option'))].sort_values(by=['Date', 'Strike Price'])
        
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
        
        return trades

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
        current_under = None
        current_id = None
        strategy_actions = {}
        position_actions = {}
        self.trades['strategy_id'] = np.nan
        last_id = 0

        for _, row in self.trades.iterrows():
            if row['Open or Close']=='OPEN': 
                # The first action in this trade is open a position -> new strategy
                # It's important to sort the trades by CLOSE-OPEN order
                if (current_date != row['Date']):
                    current_date = row['Date']
                    current_under = row['Underlying Symbol']
                    last_id += 1
                    current_id = last_id

                strategy_actions, position_actions = self._add_action(row,
                                                                position_actions,
                                                                strategy_actions,
                                                                current_id)
            else: # CLOSE
                # It's a new action of same trade
                # If the first action is a CLOSE, it's a adjustment probably
                if (current_date != row['Date']) or (current_under != row['Underlying Symbol']):
                    current_date = row['Date']
                    current_under = row['Underlying Symbol']
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
        df = pd.DataFrame.from_records(actions, columns=Action._fields).sort_values(by=['strategy_id', 'date', 'open_close', 'strike'])
        return df


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
        value = sum([l.value for l in legs])

        if strategy_name == 'Put Credit Spread' or strategy_name == 'Call Credit Spread':
            max_loss = abs(legs[1].strike - legs[0].strike) * legs[0].quantity * multiplier
            #print(strategy_name, value, max_loss, legs[0].quantity,  (max_loss-value)/legs[0].quantity)
            max_loss -= value
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
        strategies = self.trades.copy()       
        for _, strategy_trades in self.trades.groupby(['strategy_id']):
            open_legs = []
            strategy_created = False
            dte = None
            open_positions = 0
            closed_positions = 0

            for _, row in strategy_trades.iterrows():
                if (row['open_close']=='OPEN'):
                    open_positions += 1 
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

                else:
                    closed_positions += 1
                    strategy_created = True

            if closed_positions == open_positions:
                strategy_closed = True
            else:
                strategy_closed = False

            name = self._strategy_pattern(open_legs, dte)
            max_loss, max_profit, limit = self._strategy_measures(name, open_legs)

            strategies.loc[strategy_trades.index.values, 'strategy_name'] = name
            strategies.loc[strategy_trades.index.values, 'max_loss'] = max_loss
            strategies.loc[strategy_trades.index.values, 'max_profit'] = max_profit
            strategies.loc[strategy_trades.index.values, 'dte'] = dte
            strategies.loc[strategy_trades.index.values, 'strategy_type'] = strategy_type
            strategies.loc[strategy_trades.index.values, 'strategy_state'] = 'CLOSED' if strategy_closed else 'OPEN'
            strategies.loc[strategy_trades.index.values, 'direction'] = STRATEGIES[name]['direction']
            strategies.loc[strategy_trades.index.values, 'limit'] = limit

        strategies = strategies.groupby(['strategy_id',
                            'strategy_name',
                            'underlying',
                            'strategy_state']).agg(open_datetime=('date', np.min),
                                                close_datetime=('date', np.max),
                                                quantity=('quantity', np.max),
                                                expiration_date=('expiration', np.max),
                                                value=('value', np.sum),
                                                commissions=('commissions', np.sum),
                                                fees=('fees', np.sum),
                                                max_profit=('max_profit', np.max),
                                                max_loss=('max_loss', np.max),
                                                dte=('dte', np.max),
                                                strategy_type=('strategy_type', np.max),
                                                direction=('direction', np.max),
                                                limit=('limit', np.max))

        strategies['result'] = strategies['value'] + strategies['commissions'] + strategies['fees']
        strategies['result_position'] = strategies['result'] / strategies['quantity']

        strategies['days'] = (strategies['close_datetime'] - strategies['open_datetime']).dt.days + 1
        strategies['result_day_position'] = strategies['result'] / strategies['days'] / strategies['quantity']

        strategies['max_profit_dte_position'] = strategies['max_profit'] / strategies['dte'] / strategies['quantity']
        strategies['max_profit_day_position'] = strategies['max_profit'] / strategies['days'] / strategies['quantity']
        strategies['max_loss_max_profit'] = strategies['max_loss'] / strategies['max_profit']
        strategies['max_loss_pct'] = strategies['max_loss'] / self.account_size
        strategies['comfee_value'] = (strategies['commissions'] + strategies['fees']) / strategies['value']
        return strategies.reset_index()

    def closed_strategies(self):
        closed_s = self.strategies[self.strategies['strategy_state']=='CLOSED'].copy()
        closed_s['close_date'] = closed_s['close_datetime'].dt.tz_convert(tz=None) # remove tz info
        closed_s['date_grp_w'] = closed_s['close_date'].dt.year.astype(str) + ' ' + closed_s['close_date'].dt.isocalendar().week.astype(str)
        closed_s['close_date'] = closed_s['close_date'].dt.date # I can convert it because there aren't NA values
        return closed_s

    def closed_strategies_daily(self):
        closed_s = self.closed_strategies()
        min_date = closed_s['close_date'].min()
        max_date = closed_s['close_date'].max()

        idx = pd.date_range(start=min_date, end=max_date, normalize=True) # Daily index
        closed_s_d = closed_s.copy().reset_index().set_index(['close_date']) # Set the date as index
        closed_s_d = pd.merge(idx.to_series(name='date_idx'),closed_s_d, how='left', left_index=True, right_index=True)
        closed_s_d['date_grp_w'] = closed_s_d['date_idx'].dt.year.astype(str) + ' ' + closed_s_d['date_idx'].dt.isocalendar().week.astype(str).str.pad(2,fillchar='0')
        return closed_s_d

    def open_strategies(self):
        open_s = self.strategies[self.strategies['strategy_state']=='OPEN'].copy()
        open_s['date_grp_m'] = open_s['expiration_date'].dt.year.astype(str) + ' ' + open_s['expiration_date'].dt.month.astype(str).str.pad(2,fillchar='0')
        open_s['date_grp_w'] = open_s['expiration_date'].dt.year.astype(str) + ' ' + open_s['expiration_date'].dt.isocalendar().week.astype(str).str.pad(2,fillchar='0')
        return open_s

    def closed_strategies_summary(self, strategy_name, strategy_type, account_size, from_date, verbose=False):
        closed_s = self.closed_strategies()
        closed_s = closed_s[(closed_s['strategy_name'] == strategy_name)
                             & (closed_s['strategy_type']==strategy_type)
                             & (closed_s['close_date']>=from_date)]

        if verbose:
            print(closed_s[['close_date', 'underlying', 'result', 'value', 'max_loss', 'quantity']].sort_values(by=['close_date', 'underlying']))

        total_return = closed_s[['result']].sum().squeeze()
        total_positions = closed_s[['quantity']].sum().squeeze()
        win_positions = closed_s[closed_s['result']>=0][['quantity']].sum().squeeze()
        loss_positions = total_positions - win_positions
        win_ratio = win_positions/total_positions
        loss_ratio = loss_positions/total_positions
        mean_result_win_positions = closed_s[closed_s['result']>=0][['result']].sum().squeeze() / win_positions
        mean_max_loss = closed_s['max_loss'].sum().squeeze() / total_positions
        mean_open_days = closed_s['days'].mean()
        
        if loss_positions:
            mean_result_loss_positions = closed_s[closed_s['result']<0][['result']].sum().squeeze() / loss_positions
        else:
            mean_result_loss_positions = 0

        win_return = mean_result_win_positions / mean_max_loss
        loss_return = mean_result_loss_positions / mean_max_loss

        expected_value = mean_result_win_positions * win_ratio + mean_result_loss_positions * loss_ratio
        expected_return = win_return * win_ratio - abs(loss_return) * loss_ratio
        annualized_expected_return = pow(expected_return + 1, 365/mean_open_days) - 1
        portfolio_expected_return = expected_return * (mean_max_loss/account_size)
        portfolio_annualized_expected_return = pow(portfolio_expected_return + 1, 365/mean_open_days) - 1

        print(f'\n{strategy_type.capitalize()} {strategy_name} \n')
        print(f'Account size ${account_size}')
        print(f'Total return: ${total_return:.0f}')
        print(f'Rate of return (RoR): {(total_return/self.account_size):.2%}')
        print(f'Average size position: ${mean_max_loss:.0f}')
        print(f'Average open days: {mean_open_days:.0f}')
        print(f'Total positions: {total_positions:.0f}')
        print(f'Win positions: {win_positions:.0f}')
        print(f'Loss positions: {loss_positions:.0f}')
        print(f'Win ratio: {win_ratio:.2%}')
        print(f'Loss ratio: {loss_ratio:.2%}')
        print(f'Average win position result: ${mean_result_win_positions:.2f}')
        print(f'Average loss position result: ${mean_result_loss_positions:.2f}')
        print(f'\nExpected value: ${expected_value:.2f}')
        print(f'Expected return: {expected_return:.2%}')
        print(f'Annualized expected return: {annualized_expected_return:.2%}')
        print(f'Portfolio expected return: {portfolio_expected_return:.2%}')      
        print(f'Portfolio annualized expected return: {portfolio_annualized_expected_return:.2%}')    
    
    def open_strategies_summary(self):
        open_s = self.open_strategies()
        total_positions = open_s[['quantity']].sum().squeeze()
        
        print(f'Total open positions: {total_positions:.0f}')
