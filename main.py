import datetime
from pathlib import Path
from pytaco.tastyworks import TWAccount

FILE_NAME = 'data/tw.csv'

f = Path.cwd().joinpath(FILE_NAME)

if f.exists():
    tw = TWAccount(f)
    tw._to_trades()
else:
    print("The file doesn't exist")


