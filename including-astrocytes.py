from brian2 import * 
prefs.codegen.target = 'weave'

from time import time
import itertools
import Graphics as artist 
import numpy as np
from scipy.linalg import block_diag

from matplotlib import rcParams
rcParams['text.usetex'] = True

Eex=0*mV
Ein=-80*mV
Vrest=-60*mV
tau=20*ms
tauex=5*ms
tauin=10*ms
Vth=-40*mV
Ib=0.04*nA


gj_weights = {'low':0.02,'medium':0.1,'high':0.2}
AN_j_weights = gj_weights.items() #deep copy 

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)

def make_random_synapses(n,p):
	isSynapse = np.random.random_sample(size=(n,n))
	isSynapse[isSynapse<p] = 1
	isSynapse[isSynapse!=1]=0

	return np.nonzero(isSynapse)

def make_synfire_connections(neurons_per_chain,n_chains):
	block = np.ones((neurons_per_chain,neurons_per_chain))
	sequence = [block] * (n_chains-1)
	res =  np.fliplr(block_diag(*sequence))
	res = np.r_[np.c_[np.zeros((neurons_per_chain*(n_chains-1),neurons_per_chain)),res],np.zeros((neurons_per_chain,total_neurons))]
	return np.flipud(res)

start = time()
total_neurons = 100
n_chains = 10
#Could explore embedding in a network
neurons_per_chain = total_neurons/float(n_chains)
model = Equations('''
	dV/dt = (Vrest-V+gex*(Eex-V)+gin*(Ein-V) + Igap + gastro*Iastro)*(1./tau) : volt
	dgin/dt = -gin*(1./tauin) : 1
	dgex/dt = -gex*(1./tauex) : 1
	dgastro/dt = -gastro*(1./(3*tau)) : 1
	Igap :volt 
	Iastro: volt
	''')

we = 2 # excitatory synaptic weight
wi = 1.5 # inhibitory synaptic weight
wastro = 1
wneuron = 1
#Astrocyte- neural model

astrocyte_model = '''
	dV/dt = (Vrest-V + gneuron*Ineuron)*(1./tau) : volt
	Ineuron : volt 
	dgneuron/dt = - gneuron*(1./(3*tau)) : 1
'''
#Excitatory and inhibitory conductances are in multiples of gleak
N = NeuronGroup(total_neurons,model=model,threshold='V>Vth', reset='V=Vrest',refractory=5*ms)
A = NeuronGroup(1,model=astrocyte_model,threshold='V>Vrest',reset='V=Vrest')

Ce = Synapses(N, N,pre='gex += we')
Ci = Synapses(N, N,pre='gin += wi')

#Random background

excitatory_from,excitatory_to = make_random_synapses(total_neurons,p=0.1)
inhibitory_from,inhibitory_to = make_random_synapses(total_neurons,p=0.2)

Ce.connect(excitatory_from,excitatory_to)
Ci.connect(inhibitory_from,inhibitory_to) #Reflecting twice as many inhibitory as excitatory neurons

#Synfire chain
chain_idx = reshape(arange(total_neurons),(-1,n_chains))
synfire_connection = Synapses(N,N,pre='gex+=(1.1*we)')
for previous,next in pairwise(chain_idx):
	for neuron in previous:
		synfire_connection.connect(neuron,next)

#Gap Junctions
gap_junctions = Synapses(N, N, '''
             w : 1 # gap junction conductance
             Igap_post = w * (V_pre - V_post) : volt (summed)
             ''')
gap_junctions.connect(True)
gap_junctions.w = .02

#Astrocyte connections

AN_junctions = Synapses(A,N,'''
	w : 1 # gap junction conductance
	Iastro_post = w*(V_pre-V_post) : volt (summed)
	''')

AN_junctions.connect(True)
AN_junctions.w = 0.1*(1/float(total_neurons)) #Gain simulates buffering?

NA_junctions = Synapses(N,A,'''
	w : 1 #gap junction conductance
	Ineuron_post = w*V_pre : volt (summed)
''')
NA_junctions.connect(True)
NA_junctions. w = 0.02

#Monitors
M = StateMonitor(N,('V','Iastro'),record=True)
R = SpikeMonitor(N)

N.V = Vrest
duration = 4000*ms
#--Baseline
run(duration/8,report='text')

#Stimulate
N.V[chain_idx[0,:]] = -20*mV
run(duration/8)

#Stimulate and block gap junctions
N.V[chain_idx[0,:]] = -20*mV
gap_junctions.w = 0
run(duration/8)

#--Interval
#Stimulate, gap junctions unblocked
N.V[chain_idx[0,:]] = -20*mV
gap_junctions.w = 0.02
run(duration/8)

#--Stimulate and block astrocyte interaction
N.V[chain_idx[0,:]] = -20*mV
NA_junctions.w=0
AN_junctions.w=0
run(duration/8)

#--Interval
N.V[chain_idx[0,:]] = -20*mV
NA_junctions.w=0.02
AN_junctions.w = 0.01*(1/float(total_neurons))
run(duration/8)

#--Block both types of gap junctions
N.V[chain_idx[0,:]] = -20*mV
NA_junctions.w=0
AN_junctions.w=0
gap_junctions.w = 0
run(duration/8)

#--Interval to allow for recovery
N.V[chain_idx[0,:]] = -20*mV
gap_junctions.w=0.02
NA_junctions.w=0.02
AN_junctions.w = 0.01*(1/float(total_neurons))
run(duration/8)
plot(R.t/ms,R.i,'k.',clip_on=False)
ax = plt.gca()

for idx in chain_idx[:,-1]:
	ax.axhline(y=idx,color='k',linestyle='--')
for partition in np.arange(0,duration/ms,500):
	ax.axvline(x=partition,color='r',linestyle='-')

ax.set_xlim(xmin=0,xmax=duration/ms)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Neuron')

'''
Ce_connections = np.zeros((total_neurons,total_neurons))
Ce_connections[excitatory_from,excitatory_to] = 1
fig = plt.figure()
connection_panel = fig.add_subplot(111)
connection_strengths = make_synfire_connections(neurons_per_chain,n_chains)*we+Ce_connections+0.02*np.flipud(np.fliplr(np.eye(total_neurons,k=-1)))
cax = connection_panel.imshow(connection_strengths,interpolation='nearest',aspect='auto')
plt.colorbar(cax)
artist.adjust_spines(connection_panel)
'''
labels = ['','','Neuron','','Astrocyte','','Both','']
for i,label in enumerate(labels):
	ax.annotate(r'\textsc{\textbf{%s}}'%label,xy=(float(i)/len(labels),.05),xycoords='axes fraction', 
		clip_on=False)
artist.adjust_spines(ax)
tight_layout()
'''
raster = np.zeros((total_neurons,duration/ms))
for neuron,timestamp in zip(R.i,R.t/ms):
	raster[neuron,timestamp] = 1
raster = np.flipud(raster)
overlap = raster.T.dot(raster[:,int(duration/8/ms):int(2*duration/8/ms)])
overlap /= (duration/8/ms) #i.e.length of window
figure()
plot(overlap,'k.-')
ax = plt.gca()
artist.adjust_spines(ax)
ax.set_ylabel(r'\textbf{\textsc{Overlap}}')
ax.set_xlabel(r'\textbf{\textsc{Time (ms)}}')
tight_layout()
'''
print time()-start,' duration'
show()