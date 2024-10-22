import sys
sys.path.append('../learner/baselines/')
from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (6, 5),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
tb_to_plot = ["spaceinvaders_novice-ppo-2", "spaceinvaders_rl_ppo", "spaceinvaders_livelong-ppo-2", "spaceinvaders_random_ppo"]
labels = ["LfL", "RL", "LiveLong", "Random"]
linestyles = ["-", "--", "-.", ":"]
colors = ['r','g','b','k']
num_steps = 2475
for cnt, data_dir in enumerate(tb_to_plot):
    print(cnt, data_dir)
    results = pu.load_results('~/logs/' + data_dir) 
    #print(results) 
    r = results[0]
    #print(r)
    #print(r.progress.total_timesteps)
    #plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(np.cumsum(r.monitor.l)[:num_steps], pu.smooth(r.monitor.r, radius=100)[:num_steps], label=labels[cnt], linestyle=linestyles[cnt], color = colors[cnt], linewidth=2)
#    plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
    #pu.plot_results(results)

plt.xlabel("steps")
plt.ylabel("return")
plt.legend()
plt.tight_layout()
plt.savefig("spaceinvaders_ablation.png")
plt.show()

