import csv
import Graphics as artist
import matplotlib.pyplot as plt 

from matplotlib import rcParams

rcParams['text.usetex'] = True

data = sorted(list(csv.DictReader(open('record'),delimiter=' ')),key=lambda item:item['ggap'])
#Everything in this is a string


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter([float(datum['ggap']) for datum in data],[float(datum['sigma']) for datum in data],
	c='k',s=20,clip_on=False)
artist.adjust_spines(ax)
ax.set_xlabel(r'$g_{gap}$')
ax.set_ylabel(r'$\sigma$',rotation='horizontal')
plt.show()