import pandas as pd
import matplotlib.pyplot as plt

clean_file_names=["art_daily_no_noise.csv","art_daily_perfect_square_wave.csv", 
                  "art_daily_small_noise.csv",'art_flatline.csv',"art_noisy.csv"]

anomaly_file_names=["art_daily_flatmiddle.csv", "art_daily_jumpsdown.csv",
                    "art_daily_jumpsup.csv","art_daily_nojump.csv", 
                    "art_increase_spike_density.csv", "art_load_balancer_spikes.csv"]

fig,axs=plt.subplots(3,2)
x=0
y=0
for f in clean_file_names:
    df=pd.read_csv(f)
    axs[x, y].plot(df['timestamp'],df['value'])
    axs[x, y].set_title(f)
    if y==0:
        y=1
    else:
        x+=1
        y=0

#graph
fig.tight_layout()
plt.show()

fig,axs=plt.subplots(3,2)
x=0
y=0
for f in anomaly_file_names:
    df=pd.read_csv(f)
    axs[x, y].plot(df['timestamp'],df['value'])
    axs[x, y].set_title(f)
    if y<1:
        y+=1
    else:
        x+=1
        y=0

#graph
fig.tight_layout()
plt.show()