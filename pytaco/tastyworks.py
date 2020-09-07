from pathlib import Path
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
        