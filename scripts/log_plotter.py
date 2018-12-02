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
tb_to_plot = ["pong_novice-ppo-2", "pong_rl_ppo", "pong_livelong-ppo-2", "pong_random_ppo"]
labels = ["LfL", "RL", "LiveLong", "Random"]
linestyles = ["-", "--", "-.", ":"]
colors = ['r','g','b','k']
num_steps = 1000
for cnt, data_dir in enumerate(tb_to_plot):
    results = pu.load_results('~/logs/' + data_dir) 
    
    r = results[0]
    print(r.progress.total_timesteps)
    #plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
    plt.plot(r.progress.total_timesteps[:num_steps], r.progress.eprewmean[:num_steps], label=labels[cnt], linestyle=linestyles[cnt], color = colors[cnt])
    #pu.plot_results(results)

plt.xlabel("steps")
plt.ylabel("return")
plt.legend()
plt.tight_layout()
#plt.savefig()
plt.show()

