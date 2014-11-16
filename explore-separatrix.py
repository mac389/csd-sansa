#import brian_no_units
from brian import *
from scipy.optimize import fsolve
import time

from brian.library.IF import *
from brian.library.synapses import *

from sys import argv



N=1000

Vr = -70 * mV
Vt = -55 * mV
taum = 10 * ms
taupsp = 0.325 * ms
weight =  4.86 * mV

tau = 20 * ms
sigma = 5 * mV
frac = float(argv[1])/100
g_gap = frac*(1. / N)
beta = 60 * mV * 2 * ms
delta = Vt - Vr

def findIntersection(fun1,fun2,x0):
 return fsolve(lambda x : fun1(x) - fun2(x),x0)

def myreset(P, spikes):
    P.V[spikes] = Vr # reset
    P.V += g_gap * beta * len(spikes) # spike effect
    P.u -= delta * len(spikes)


def minimal_example():
    # Neuron model parameters
    Vr = -70 * mV
    Vt = -55 * mV
    taum = 10 * ms
    taupsp = 0.325 * ms
    weight = 4*mV#4.86 * mV
    # Neuron model
    equations = Equations('''
        dV/dt = (-(V-Vr)+x+g_gap*(u-N*V))*(1./taum)                            : volt
        dx/dt = (-x+y)*(1./taupsp)                               : volt
        dy/dt = -y*(1./taupsp)+25.27*mV/ms+(39.24*mV/ms**0.5)*xi : volt
        du/dt = (N*Vr-u)/tau : volt # input from other neurons
        ''')

    # In this model, gap junctions between neurons are one giant voltage sync
    # Similarly, gap junctions between astrocytes are one giant excitatory current sync
    # both use same formalism
    # Neuron groups
    P = NeuronGroup(N=10000, model=equations,
                  threshold=Vt, reset=Vr)
#    P = NeuronGroup(N=1000, model=(dV,dx,dy),init=(0*volt,0*volt,0*volt),
#                  threshold=Vt,reset=Vr,refractory=1*ms)

    Pinput = PulsePacket(t=50 * ms, n=85, sigma=1 * ms)
    # The network structure
    Pgp = [ P.subgroup(100) for i in range(100)]
    C = Connection(P, P, 'y')
    for i in range(9):
        C.connect_full(Pgp[i], Pgp[i + 1], weight)
    Cinput = Connection(Pinput, P, 'y')
    Cinput.connect_full(Pinput, Pgp[0], weight)
    # Record the spikes
    Mgp = [SpikeMonitor(p, record=True) for p in Pgp]
    Minput = SpikeMonitor(Pinput, record=True)
    monitors = [Minput] + Mgp
    # Setup the network, and run it
    P.V = Vr + rand(len(P)) * (Vt - Vr)
    run(100 * ms)
    # Plot result
    raster_plot(showgrouplines=True, *monitors)
    show()


# DEFAULT PARAMATERS FOR SYNFIRE CHAIN
# Approximates those in Diesman et al. 1999
model_params = Parameters(
    # Simulation parameters
    dt=0.1 * ms,
    duration=100 * ms,
    # Neuron model parameters
    taum=10 * ms,
    taupsp=0.325 * ms,
    Vt= -55 * mV,
    Vr= -70 * mV,
    abs_refrac=1 * ms,
    we=34.7143,
    wi= -34.7143,
    psp_peak=0.14 * mV,
    # Noise parameters
    noise_neurons=20000,
    noise_exc=0.88,
    noise_inh=0.12,
    noise_exc_rate=2 * Hz,
    noise_inh_rate=12.5 * Hz,
    tau = 20 * ms,
    sigma = 5 * mV,
    g_gap = 0*(1. / N),
    beta = 60 * mV * 2 * ms,
    delta = Vt - Vr,
    computed_model_parameters="""
    noise_mu = noise_neurons * (noise_exc * noise_exc_rate - noise_inh * noise_inh_rate ) * psp_peak * we
    noise_sigma = (noise_neurons * (noise_exc * noise_exc_rate + noise_inh * noise_inh_rate ))**.5 * psp_peak * we
    """
    )

# MODEL FOR SYNFIRE CHAIN
# Excitatory PSPs only
def Model(p):
    equations = Equations('''
        dV/dt = (-(V-Vr)+x+g_gap*(u-N*V))*(1./taum)                            : volt
        dx/dt = (-x+y)*(1./taupsp)                               : volt
        dy/dt = -y*(1./taupsp)+25.27*mV/ms+(39.24*mV/ms**0.5)*xi : volt
        du/dt = (N*Vr-u)/tau : volt # input from other neurons
        ''')

    return Parameters(model=equations, threshold=p.Vt, reset=myreset)

default_params = Parameters(
    # Network parameters
    num_layers=50,
    neurons_per_layer=100,
    neurons_in_input_layer=100,
    # Initiating burst parameters
    initial_burst_t=50 * ms,
    initial_burst_a=85,
    initial_burst_sigma=1 * ms,
    # these values are recomputed whenever another value changes
    computed_network_parameters="""
    total_neurons = neurons_per_layer * num_layers
    """,
    # plus we also use the default model parameters
    ** model_params
    )

# DEFAULT NETWORK STRUCTURE
# Single input layer, multiple chained layers
class DefaultNetwork(Network):
    def __init__(self, p):
        # define groups
        chaingroup = NeuronGroup(N=p.total_neurons, **Model(p))
        inputgroup = PulsePacket(p.initial_burst_t, p.neurons_in_input_layer, p.initial_burst_sigma)
        layer = [ chaingroup.subgroup(p.neurons_per_layer) for i in range(p.num_layers) ]
        # connections
        chainconnect = Connection(chaingroup, chaingroup, 2)
        for i in range(p.num_layers - 1):
            chainconnect.connect_full(layer[i], layer[i + 1], p.psp_peak * p.we)
        inputconnect = Connection(inputgroup, chaingroup, 2)
        inputconnect.connect_full(inputgroup, layer[0], p.psp_peak * p.we)
        # monitors
        chainmon = [SpikeMonitor(g, True) for g in layer]
        inputmon = SpikeMonitor(inputgroup, True)
        mon = [inputmon] + chainmon
        # network
        Network.__init__(self, chaingroup, inputgroup, chainconnect, inputconnect, mon)
        # add additional attributes to self
        self.mon = mon
        self.inputgroup = inputgroup
        self.chaingroup = chaingroup
        self.layer = layer
        self.params = p

    def prepare(self):
        Network.prepare(self)
        self.reinit()

    def reinit(self, p=None):
        Network.reinit(self)
        q = self.params
        if p is None: p = q
        self.inputgroup.generate(p.initial_burst_t, p.initial_burst_a, p.initial_burst_sigma)
        self.chaingroup.V = q.Vr + rand(len(self.chaingroup)) * (q.Vt - q.Vr)

    def run(self):
        Network.run(self, self.params.duration)

    def plot(self):
        raster_plot(ylabel="Layer", title="Synfire chain raster plot",
                   color=(1, 0, 0), markersize=3,
                   showgrouplines=True, spacebetweengroups=0.2, grouplinecol=(0.5, 0.5, 0.5),
                   *self.mon)

def estimate_params(mon, time_est):
    # Quick and dirty algorithm for the moment, for a more decent algorithm
    # use leastsq algorithm from scipy.optimize.minpack to fit const+Gaussian
    # http://www.scipy.org/doc/api_docs/SciPy.optimize.minpack.html#leastsq
    i, times = zip(*mon.spikes)
    times = array(times)
    times = times[abs(times - time_est) < 15 * ms]
    if len(times) == 0:
        return (0, 0 * ms)
    better_time_est = times.mean()
    times = times[abs(times - time_est) < 5 * ms]
    if len(times) == 0:
        return (0, 0 * ms)
    return (len(times), times.std())

def single_sfc():
    net = DefaultNetwork(default_params)
    net.run()
    #net.plot()

def state_space(grid, neuron_multiply, verbose=True):
    amin = 0
    amax = 100
    sigmamin = 0. * ms
    sigmamax = 3. * ms

    params = default_params()
    params.num_layers = 1
    params.neurons_per_layer = params.neurons_per_layer * neuron_multiply

    net = DefaultNetwork(params)

    i = 0
    # uncomment these 2 lines for TeX labels
    #import pylab
    #pylab.rc_params.update({'text.usetex': True})
    if verbose:
        print "Completed:"
    start_time = time.time()
    figure()
    param_list = []
    for ai in range(grid + 1):
        for sigmai in range(grid + 1):
            a = int(amin + (ai * (amax - amin)) / grid)
            if a > amax: a = amax
            sigma = sigmamin + sigmai * (sigmamax - sigmamin) / grid
            params.initial_burst_a, params.initial_burst_sigma = a, sigma
            net.reinit(params)
            net.run()
            (newa, newsigma) = estimate_params(net.mon[-1], params.initial_burst_t)
            newa = float(newa) / float(neuron_multiply)
            col = (float(ai) / float(grid), float(sigmai) / float(grid), 0.5)
            yslope = newa-a
            xslope = np.diff([newsigma/ms,sigma/ms])[0]
            param_list.append(tuple((a,sigma,yslope,xslope)))
            plot([sigma / ms, newsigma / ms], [a, newa], color=col)
            #print [sigma / ms, newsigma / ms], [a, newa]
            plot([sigma / ms], [a], marker='.', color=col, markersize=15)
            i += 1
            if verbose:
                print str(int(100. * float(i) / float((grid + 1) ** 2))) + "%",
        if verbose:
            print
    if verbose:
        print "Evaluation time:", time.time() - start_time, "seconds"
    xlabel(r'$\sigma$ (ms)')
    ylabel('a')
    title('Synfire chain state space')
    axis([sigmamin / ms, sigmamax / ms, amin, amax])
    #colorbar()
    return param_list

#minimal_example()
print 'Computing SFC with multiple layers'
single_sfc()
#print 'Plotting SFC state space'
#state_space(3,1)
params = state_space(20,10)

#separatrix = [(a,sigma) for a,sigma,yslope,xslope in params if 
 #       float(yslope)/xslope]

a,sigma,yslope,xslope = zip(*params)
with open('record-astrocytes','a') as f:
            
    try:
        #a-nullcline
        a_nullcline_idx = (np.diff(np.sign(np.array(yslope))) != 0)*1
        a_nullcline_coords_x = 1000*np.array(sigma)[a_nullcline_idx==1]
        a_nullcline_coords_y = np.array(a)[a_nullcline_idx==1]

        a_nullcline_coords = zip(a_nullcline_coords_x,a_nullcline_coords_y)
        a_nullcline_coords = [(x,y) for x,y in a_nullcline_coords if (x>0)*(x<3.0) and (y>0)*(y<100)]

        a_nullcline_coords_x,a_nullcline_coords_y = zip(*a_nullcline_coords)

        #s-nullcline
        s_nullcline_idx = (np.diff(np.sign(np.array(xslope))) != 0)*1
        s_nullcline_coords_x = 1000*np.array(sigma)[s_nullcline_idx==1]
        s_nullcline_coords_y = np.array(a)[s_nullcline_idx==1]

        s_nullcline_coords = zip(s_nullcline_coords_x,s_nullcline_coords_y)
        s_nullcline_coords = [(x,y) for x,y in s_nullcline_coords if (x>0)*(x<3.0) and (y>0)*(y<100)]

        s_nullcline_coords_x,s_nullcline_coords_y = zip(*s_nullcline_coords)


        #idx = (np.diff(np.sign(np.array(yslope)/np.array(xslope))) !=0)*1
        #print idx
        #print = (np.diff(np.sign(xslope)) != 0)*1
        hold(True)
        plot(a_nullcline_coords_x,a_nullcline_coords_y,'k--',linewidth=2)
        plot(s_nullcline_coords_x,s_nullcline_coords_y,'k-.',linewidth=2)
        #savefig('separatrix-%d.png'%(frac*10),dpi=300)


        #calculate point of intersection of nullclines
        s_fxn = np.poly1d(np.polyfit(s_nullcline_coords_x,s_nullcline_coords_y,2))
        a_fxn = np.poly1d(np.polyfit(a_nullcline_coords_x,a_nullcline_coords_y,2))


        intersection=findIntersection(s_fxn,a_fxn,0.2)
        print>>f, intersection[0], a_fxn(intersection)[0], frac
    except:
        print>>f, '0', '0', frac