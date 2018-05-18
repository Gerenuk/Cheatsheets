Visualization of data
=====================

Categorical-Numerical data
--------------------------
You can use Violin plot or Bean plots (violin+scatter):
http://statsmodels.sourceforge.net/stable/generated/statsmodels.graphics.boxplots.violinplot.html
http://statsmodels.sourceforge.net/devel/generated/statsmodels.graphics.boxplots.beanplot.html
http://www.cbs.dtu.dk/~eklund/beeswarm/

Numerical (1D)
--------------
Kernel density estimation
.........................
http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html:

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt
    
    X = <data>
    kde = stats.gaussian_kde(X)
    X_kde = np.linspace(min(X), max(X), 100)
    plt.plot(X, kde(X_kde))
    
Plots
=====

Density estimation
------------------

          from scipy.stats import gaussian_kde
          d=sorted(df.VAR.dropna())
          dens=gaussian_kde(d, bw_method=0.1) # omit bw_method for auto
          plt.plot(d, dens(d))
          
          
Histogram
---------
numpy?

http://matplotlib.org/examples/pylab_examples/histogram_demo_extended.html
plt.hist(DATA, bins=30)


import statsmodels.api as sm
sm.graphics.violinplot([data[data.varclass == c].var for c in classes], ax=plt)

For histogram in log scale you might need the bottom parameter::
    
    plt.hist(..., bottom=0.1)
    plt.set_yscale("log")
    
Bar plot
--------
::
    f=plt.figure()
    ax=f.subplot(111)
    ax.bar(x, y, align="center")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
          
Matplotlib plot
---------------
Figure
......
::
    fig = plt.figure(figsize=(8,4), dpi=100)   # make 800x600 figure
    fig, axes = plt.subplots(figsize=(8,4), dpi=100)   # make 800x600 figure
    fit.savefig("file.pdf")  # PDF recommended

Axis scale
..........
ax.semilogx()
ax.semilogy()
ax.loglog()
ax.set_aspect("equal")
ax.set_yscale("log")

(or ax.set_xscale(), ax.xscale() )

ax.set_title("Title")
ax.set_ylabel("Text", fontsize=30)
ax.yaxis.label.set_size(30)

plt.semilogx()
plt.xlabel("Preis")
plt.grid(True, "both")
plt.ylabel("Rang")

Subplots
........
::
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes[1,1].plot(...)
    
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for ax in axes:
       ax.plot(...)
    fig.tight_layout()   # adjust to avoid overlapping labels
    
    plt.subplot2grid(...)  # for layouts with span
    matplotlib.gridspec.GridSpec(...)   # to change subplot sizes and ratios
    
    fig.add_axes([...])  # to add axes; e.g. insets
     



For complex grids::
    
    http://matplotlib.org/users/gridspec.html

Annotate
........
::
     cb=ax.colorbar() # color bar legend
     cb.set_ticks([0, .5, 1])
     cb.set_ticklabels(['bad', 'ok', 'great!'])
     
     ax.annotate(text, xy=(x, y), horizontalalignment='center', verticalalignment='center')
     
     ax.legend(["A", "B", "C"])    # or ->
     ax.plot(..., label="A")
     ax.legend(loc=0)              # loc=0 [auto], 1 [upper right], 2, 3, 4
     
     matplotlib.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})  # global fonts; STIX recommended
     matplotlib.rcParams.update({'font.size': 18, 'text.usetex': True})  # for LaTeX
     
     from matplotlib import ticker  # see http://matplotlib.org/api/ticker_api.html
     formatter = ticker.ScalarFormatter(useMathText=True)
     formatter.set_scientific(True) 
     formatter.set_powerlimits((-1,1)) 
     ax.yaxis.set_major_formatter(formatter)   # show $.$ and put 10^$ on top
     
     rcParams['xtick.major.pad'] = 5   # space axis to label
     rcParams['ytick.major.pad'] = 5
     
     fig.subplots_adjust(left=0.15, right=.9, bottom=0.1, top=0.9)  # if labels clipped
     
     ax.grid(True)   # also has many options for style
     
     ax.spines["right"].set_color("none")  # to change style of box
     ax.spines["left"].set_position(("data",0))   # to make central 0 axis
     
     ax2 = ax.twinx()   # plot on same figure but with different scale (labels will be on right spine)
     
     ax.text(0.1, 0.1, "text", fontsize=20, color="blue")


Color
.....

http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
http://matplotlib.org/examples/color/colormaps_reference.html

from pylab import *
plt.scatter(x, y, c=[...], cmap=get_cmap("gist_rainbow"))

All matplotlib colormaps have a version with name ending "_r" for the reversed order.

Plot color with `color='b.-'` or `color="#1155dd"`
Line style with `ls=...`. For example "-." or "*"
Marker symbol with `marker=...` For example `+o*s,.1234`

Nicer plots
...........
Nicer plots can be achieved with `prettyplotlib <http://olgabot.github.io/prettyplotlib/>`_:
::
    import prettyplotlib as ppl
    ppl.plot(...)
    
See more `examples <https://github.com/olgabot/prettyplotlib/wiki/Examples-with-code>`_.

Or use
::
    from matplotlib import style
    style.use('ggplot')
    
Other plots
...........
::
    std=grouped.std()
    plt.fill_between(df.index, (df+std).values, (df-std).values, alpha=0.2)

Network graphs
--------------
Use networkx with
::
    import networkx as nx
    import matplotlib.pyplot as plt 
    G = nx.MultiDiGraph()
    G.add_edges_from([
        (1, 2),
        (2, 3),
        (3, 2),
        (2, 1),
    ])
    plt.figure(figsize=(8,8))
    nx.draw(G)
