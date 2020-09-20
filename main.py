from datetime import datetime
from pathlib import Path
from pytaco.tastyworks import TWAccount

FILE_NAME = 'data/tw.csv'

f = Path.cwd().joinpath(FILE_NAME)

if f.exists():
    tw = TWAccount(f)
else:
    print("The file doesn't exist")


tw.closed_strategies_summary(strategy_name='Put Credit Spread', 
                            strategy_type='monthly',
                            account_size=10000,
                            from_date=datetime(2020,1,1).date())

tw.closed_strategies_summary(strategy_name='Put Credit Spread', 
                            strategy_type='weekly',
                            account_size=10000,
                            from_date=datetime(2020,1,1).date())

tw.closed_strategies_summary(strategy_name='Call Credit Spread', 
                            strategy_type='monthly',
                            account_size=10000,
                            from_date=datetime(2020,1,1).date())

tw.closed_strategies_summary(strategy_name='Call Credit Spread', 
                            strategy_type='weekly',
                            account_size=10000,
                            from_date=datetime(2020,1,1).date())
#tw.open_strategies_summary()