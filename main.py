import datetime
from pathlib import Path
from pytaco.tastyworks import TWAccount

FILE_NAME = 'data/tw.csv'

f = Path.cwd().joinpath(FILE_NAME)

if f.exists():
    tw = TWAccount(f)
else:
    print("The file doesn't exist")

print(len(tw.data))
print(len(tw.trades))
print(tw.closed_strategies_daily())
print(tw.open_strategies())
