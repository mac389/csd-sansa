import numpy as np
import matplotlib.pyplot as plt
import Graphics as artist

from pprint import pprint
from matplotlib import rcParams
rcParams['text.usetex'] = True

N = {'e':9,'i':9}

dt = .01

M = {   'ee': 1/dt*np.kron(np.eye(int(np.sqrt(N['e'])),k=-1),np.ones((int(np.sqrt(N['e'])),int(np.sqrt(N['e'])))))
			+np.kron(np.eye(int(np.sqrt(N['e']))),np.ones((int(np.sqrt(N['e'])),int(np.sqrt(N['e'])))))
			-1/dt*np.kron(np.eye(int(np.sqrt(N['e'])),k=-2),np.ones((int(np.sqrt(N['e'])),int(np.sqrt(N['e']))))),
		'ii': np.zeros((N['i'],N['i'])),
		'ei': -np.eye(N['e'],N['i'],k=1),
		'ie': np.eye(N['i'],N['e']),
		}

#--
G = -2* np.eye((sum(N.values())))
G += np.eye(sum(N.values()),k=1)
G += np.eye(sum(N.values()),k=-1)
#G[np.triu_indices(G.shape[0],1)]=1
#G[np.triu_indices(G.shape[0],2)]=0
G[-1,0] = 1
G[0,-1] = 1
#--

duration = 1500


v = np.zeros((sum(N.values()),duration))

type_specific = True
idx = {}
if not type_specific:
	idx['e'] = np.random.choice(range(sum(N.values())),size=N['e'])
	idx['i'] = list(set(range(sum(N.values())))-set(idx['e']))
else:
	idx['e'],idx['i'] = np.array_split(range(sum(N.values())),2)


def block_gap_junctions(source,target):
	if source in ['e','i'] and target in ['e','i']:
		G[idx[target],idx[source]] = 0


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def measure_CSD(voltages,input_idx):
	peaks = np.argmax(voltages,axis=1)

	#comparison with base
	first_peak = {'time':peaks[input_idx],'amplitude':voltages[input_idx,peaks[input_idx]]}

	spread = [(peak_time-first_peak['time'],voltages[neuron,peak_time]-first_peak['amplitude'],-1 if neuron in idx['i'] else 1) 
		for neuron,peak_time in zip(range(voltages.shape[0]),peaks)]

	#determine average time between peaks
	TIME = 0
	AMPLITUDE = 1
	TYPE = 2
	
	#Aritrary threshold of 1000 ms in between peaks
	time = abs(np.average(filter(lambda time: abs(time) < 1000 and time > 0, np.diff([abs(datum[TIME]) for datum in spread]))))
	spread = len(filter(lambda time: abs(time) < 1000 and time > 0, np.diff([abs(datum[TIME]) for datum in spread])))
	print time,spread
#Initial conditions
v[:,0] = 0#np.random.random_sample(size=v[:,0].shape)

block_gap_junctions('i','i')
for t in range(1,duration):
	v[idx['e'],t] = v[idx['e'],t-1] + dt*(-v[idx['e'],t-1]+M['ee'].dot(v[idx['e'],t-1])+M['ei'].dot(v[idx['i'],t-1]))
	v[idx['i'],t] = v[idx['i'],t-1] + dt*(-v[idx['i'],t-1]+M['ii'].dot(v[idx['i'],t-1])+M['ie'].dot(v[idx['e'],t-1]))

	v[:,t] = v[:,t-1] + 10*dt*(G.dot(v[:,t-1]))

	if t>250 and t <350:
		input_idx = np.argsort(idx['e'])[:3]
		v[input_idx,t] += 0.05

heatmap_opts = {'interpolation':'nearest','aspect':'auto', 'cmap':plt.cm.binary}

topology = True
if topology:
	#Network topology
	fig,axs = plt.subplots(nrows=2,ncols=2,sharex=True,sharey=True)
	for row,source in zip(axs,['e','i']):
		for col,receiver in zip(row,['e','i']):
			designator = receiver + source
			col.imshow(M[designator],**heatmap_opts)

			artist.adjust_spines(col)

			col.set_xticks(range(N[source]))
			col.set_yticks(range(N[receiver]))

			col.set_xticklabels(range(N[source]))
			col.set_yticklabels(range(N[receiver]))

			col.set_xlabel(artist.format('Excitatory' if source=='e' else 'Inhibitory'))
			col.set_ylabel(artist.format('Inhibitory' if receiver=='i' else 'Excitatory'))

			col.set_title(r'\Large $\mathbf{M_{%s}}$'%designator)

			col.grid(True)

	fig.text(0.5,0.03,r'\Large \textbf{\underline{From}}', ha='center',va='center')
	fig.text(0.03,0.5, r'\Large \textbf{\underline{To}}', ha='center',va='center',rotation='vertical')

	fig.tight_layout()

'''
	TODO:
		1. Visualize gap junction connections
		2. Create type-speciifc gap junctions
		3. See which combination of location- and type-specific gap junctions allow the propagation of CSD
'''

#measure_CSD(v,input_idx)


gap_topology = True
if gap_topology:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(G,**heatmap_opts)

	artist.adjust_spines(ax)
	ax.set_xticks(range(sum(N.values())))
	ax.set_yticks(range(sum(N.values())))

	ax.set_xticklabels(range(sum(N.values())))
	ax.set_yticklabels(range(sum(N.values())))

	for i,label in enumerate(ax.get_xticklabels()):
		label.set_color('red' if i in idx['i'] else 'green')

	for i,label in enumerate(ax.get_yticklabels()):
		label.set_color('red' if i in idx['i'] else 'green')

	fig.tight_layout()

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(v[idx['e'][17],:])
'''
# Plot of activities
plot_opts = {'e': {'color':'k','linewidth':2,'alpha':0.8},
			 'i': {'color':'r','linewidth':2,'alpha':0.8}}
fig,axs = plt.subplots(nrows=1,ncols=2,sharex=True, sharey=True)

for panel,mode in zip(axs,['e','i']):
	for i,traces in enumerate(chunker(idx[mode],int(np.sqrt(N['e'])))):
		panel.plot(3*i+np.average(v[traces,:],axis=0),**plot_opts[mode]) 
		panel.hold(True)

	artist.adjust_spines(panel)
	panel.set_xlabel(artist.format('Time (ms)'))
	panel.set_ylabel(artist.format('Activity'))

fig.tight_layout()
plt.show()

