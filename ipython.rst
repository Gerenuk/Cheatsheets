IPython usage
=============

Startup
-------
Command line execution
......................
To start the IPython HTML Notebook run
::
   ipython3 notebook --matplotlib inline <PATH>
   
(don't use --pylab since it will pollute the namespace with numpy.all etc.)
   
Recommended imports
...................
::
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import itertools as itoo
    import collections as coll
    from reprlib import repr

Autoreload
..........
To enable `autoreload <http://ipython.org/ipython-doc/dev/config/extensions/autoreload.html>`_ use
::
    %load_ext autoreload
    %autoreload 2
    
Note that it's possible to restrict to particular modules.

The ``IPython.lib.deepreload`` module allows you to recursively reload a module: changes made to any of its dependencies will be reloaded without having to exit
::
    from IPython.lib.deepreload import reload as dreload
    
Profiles
........
To create a new profile in a custom directory use
::
    ipython3 profile create --profile-dir=./<...>
    
Interaction
-----------
To suppress text output (of e.g. a plot) use one of
::
    plt.plot(A);
    x = plt.plot(A)
    
Text
....
Text can be inserted by setting the cell to `markdown <https://github.com/adam-p/markdown-here/wiki/Markdown-Here-Cheatsheet>`_. The basic markdown is
::
    # H1
    ## H2
    *italics*
    **emphasis**
    1. list
      1. list
    * item
    [linkname](linkurl)
    `code`
    
    H1  | H2
    --- |
    A   | B
    
    $LaTeX$
    
Tables
......
Formatted HTML tables can be done with `ipy_table <http://nbviewer.ipython.org/github/epmoyer/ipy_table/blob/master/ipy_table-Reference.ipynb>`:
::
    from ipy_table import *
    make_table([[...],[...],...])
    apply_theme('basic')

Debugging
---------
http://ipython.org/ipython-doc/rel-0.13/interactive/reference.html#automatic-invocation-of-pdb-on-exceptions

Saving the notebook
-------------------
To convert as notebook use `nbconvert <http://ipython.org/ipython-doc/rel-1.0.0/interactive/nbconvert.html>`_
::
    ipython3 nbconvert --to html notebook.ipynb
    
To suppress code cells use a template with content
::
    {%- extends 'html_full.tpl' -%}
    {% block input %}
    {%- endblock input %} 
    {% block in_prompt %}
    {%- endblock in_prompt %}

and run
::
    ipython nbconvert --to html --template ./template.tpl notebook.ipynb
