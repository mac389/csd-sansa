from brian import *

tau = 20 * ms
sigma = .1*mV
N = 10
J = 1
mu = 1
v0 = 5 *mV
g_gap = 1./N
beta = 60* mV*2* ms

vt = 2*mV
vr = 0*mV
delta = vt-vr
eqs = """
dv/dt= (I-v)/tau + g_gap*(u-N*v)/tau+ sigma*xi/tau**.5 : mV
I: mV
du/dt = (N*v0-u)/tau : mV 
"""

def myreset(P, spikes):
    P.v[spikes] = vr # reset
    P.v += g_gap * beta * len(spikes) # spike effect
    P.u -= delta * len(spikes)

group = NeuronGroup(N, model=eqs, threshold=vt, reset=myreset)

C = Connection(group, group, 'v')
for i in range(N):
    C[i, (i + 1) % N] = J
#C.connect_full(group,group,weight=J)
#for i in range(N):
#    C[i,i]=0

S = SpikeMonitor(group)
trace = StateMonitor(group, 'v', record=True)

group.v = 0*mV
group.I = 0*mV

for i in range(5):
	run(20 * ms)
	group[0].I=5*mV
	run(20*ms)
	group[0].I = 0*mV
	run(90*ms)

subplot(211)
raster_plot(S)
for i in range(5):
	axvline(x=30+130*i)
subplot(212)
plot(trace.times / ms, trace[0]/mV)
show()