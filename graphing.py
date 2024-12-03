import pandas as pd
import matplotlib.pyplot as plt

pattern_changes=["art_daily_flatmiddle.csv","art_daily_jumpsdown.csv","art_daily_jumpsup.csv","art_daily_nojump.csv"]

fig,axs=plt.subplots(2,2)
x=0
y=0
for p in pattern_changes:
    df=pd.read_csv(p)
    axs[x, y].plot(df['timestamp'][2850:2850+350],df['value'][2850:2850+350])
    axs[x, y].set_title(p)
    if y==0:
        y=1
    else:
        x+=1
        y=0

plt.show()