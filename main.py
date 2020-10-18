from datetime import datetime
from pathlib import Path
from pytaco.tastyworks import TWAccount

FILE_NAME = 'data/tw.csv'

f = Path.cwd().joinpath(FILE_NAME)

if f.exists():
    tw = TWAccount(f)
else:
    print("The file doesn't exist")


closed_strategies = tw.closed_strategies()
print(closed_strategies[['strategy_id', 'underlying', 'strategy_name', 'result', 'quantity', 'strategy_type', 'result_day_position', 'close_date']].sort_values(by='close_date'))



#tw.closed_strategies_summary(strategy_name='Put Credit Spread', 
#                            strategy_type='monthly',
#                            account_size=10000,
#                            from_date=datetime(2020,1,1).date(),
#                            verbose=True)
