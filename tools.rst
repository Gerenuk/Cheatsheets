Tools for Data Mining
=====================

Deep Learning
-------------
Tensorflow
..........
* opensourced by Google
* made by engineers who also contributed to MapReduce, Theano, Pylearn2, Caffee
* can do partial subgraph computation -> distributed training
* (but currently [Nov'15] only multiple local GPU supported)
* like Theano but better(?)
* TensorBoard for debugging
* cons:
  * lack of symbolic loops ("scan" in theano)
  * currently worse in speed and memory than others
  * no Windows support

Torch
.....
* used by DeepMind

Hadoop
------
Surus
.....
https://github.com/Netflix/Surus
* Netflix
* Hive, Pig
* Techniques: Anomaly detection

BreakoutDetection
.................
https://github.com/twitter/BreakoutDetection
* Twitter


Recommended Tools
-----------------
Data Cleaning
.............
* OpenRefine: filter, histogram, rename, transform, script expressions, even parse (custom, JSON, etc), ...

Python tools
------------
* NumPy
* SciPy
* Pandas
* scikit-learn
* IPython
* patsy https://pypi.python.org/pypi/patsy/ and statsmodel http://statsmodels.sourceforge.net/ (?)
* RPy2
* Flask
* Blaze: Blaze aims to extend the structural properties of NumPy arrays to a wider variety of table and array-like structures that support commonly requested features such as missing values, type heterogeneity, and labeled arrays

GraphLab Create
...............
* backed by C++ engine
* analyze terabytes
* tabular, graphs, text, image
* SOTA ML

PuLP
....
* Linear Programming
* can generate LP files and call optimized solvers (COIN, GUROBI, ...)


Pandas
......
* weaknesses:
  * too far from metal (not low-level); not database
  * no support for memory maps (yet); HDF5/PyTables could be partial solution
  * no tight database integration; difficult on top f SQL-ish system; todo: ODBC API
  * inconsistent NA values; also needs to be frst class for analytics operations
  * RAM management difficult to understand; lots of copying
  * weak support for categorical data
  * complex groupBy gets messy
  * appending data slow; streaming data harder to deal with
  * limited type system; column metadata; units, currencies, geographic, composite, ...
  * no multicore algos
  * -> new tool "badger" (future: distributed in-memoty; multicore; ETL job building; open source)

Speed
.....
* Numba: The basic idea is by adding type casts python code, numba will transform your python byte code and feed it to the LLVM compiler make mind melting speed improvements
* PyCUDA:  PyCUDA gives you easy, Pythonic access to Nvidia‘s CUDA parallel computation API.

Visualization
.............
* Matplotlib
* MayaVi: Mayavi is a sophisticated open-source 3-D visualization for Python, produced by Enthought. It depends on some other Enthought products, which are part of the Anaconda CE distribution
* Bokeh: Bokeh (pronounced boh-Kay) is an implementation of Grammar of Graphics for Python, that also supports the customized rendering flexibility of Protovis and d3. Although it is a Python library, its primary output backend is HTML5 Canvas”. Bokeh is trying  to be ggplot for Python
* Pygal: SVG charts (with semi-automatic human-readable formatting)
* Plotly
* Seaborn

Workflow
........
* Joblib: https://pythonhosted.org/joblib/, run jobs, cache results (optimized for numpy), fast compressed persistence, easy parallelization
* Augustus: https://code.google.com/p/augustus/, represent models in PMML, do large scale
* Ruffus
* Drake

Reporting
.........
* ReportLab: http://www.reportlab.com/opensource/
* Pod: http://appyframework.org/pod.html, ODT document with Python expressions

Lacking
.......
* formula specificiation like R
* array manipulation like Octave
* ggplot for static graphics
* D3 for interactive graphics
* scalable NumPy (development by Continuum)
* embedded declarative language for data manipulation (like LINQ; syntax for complex manipulation hard in Pandas)
* IDE like RStudio

Large scale data
................
* Disco: MapReduce
* PySpark
* PyTables

Machine learning
................
* H20 and wise.io: large scale, accessible by Python
* GraphLab: distributed algo, accessible by (J)Python

Performance
...........
* NumbaPro: GPU
* PyCUDA: GPU

IDE
...
* Spyder
* PyScripter

Distros
.......
* PythonXY
* Enthought: provides scientists with a comprehensive set of tools to perform rigorous data analysis and visualization
* Anaconda: completely free enterprise-ready Python distribution for large-scale data processing, predictive analytics, and scientific computing
* Wakari: cloud-based
* PiCloud: cloud-based


Wikidpad:

Programming languages:
* Python
* R
* Java
* Scala
* Clojure

Databases:
* Hadoop, Hive, Pig
* HBase
* CouchDB
* Disco (Python) http://discoproject.org/
* PySpark (Python API for the Spark data analytics framework) http://spark.incubator.apache.org/docs/latest/api/pyspark/pyspark-module.html

Data analysis:
* Pandas http://pandas.pydata.org/
* PyTables http://pytables.github.io/
* Blaze http://continuum.io/blog/blz-format
* Scikit-learn http://scikit-learn.org/stable/
* Scikit-image http://scikit-image.org/
* Augustus (PMML producer and consumer, scoring engine) https://code.google.com/p/augustus/
* H2O http://0xdata.github.io/h2o/
* WiseIO http://about.wise.io/
* GraphLab http://select.cs.cmu.edu/code/graphlab/java_jython.html
* PyCUDA https://developer.nvidia.com/pycuda
* NumbaPro http://docs.continuum.io/numbapro/

Visualization:
* D3
* Flare
* HighCharts
* AmCharts
* matplotlib http://matplotlib.org/
* Bokeh (Continuum Analytics, interactive web-plotting) https://github.com/ContinuumIO/Bokeh
* Nodebox (programmers use Python code to create sketches and interactive visualizations) http://nodebox.net/

Python specific:
from http://strata.oreilly.com/2013/03/python-data-tools-just-keep-getting-better.html

Package stacks:
* Anaconda (Continuum Analytics) https://store.continuum.io/cshop/anaconda/
* Wakari (Continuum Analytics, cloud-based) https://www.wakari.io/
* PiCloud (cloud-computing platform) http://docs.picloud.com/howto/pyscientifictools.html

Comparison Shogun vs others (https://docs.google.com/spreadsheet/ccc?key=0Aunb9cCVAP6NdDVBMzY1TjdPcmx4ei1EeUZNNGtKUHc&hl=en#gid=0):
* Shogun has more than Sklearn:
  * Structured Output Learning
  * Large Scale Learning
  * Multitask Learning
  * Domain Adaptation
  * many programming language bindings
  * more SVM solvers
  * multiple kernel learning (MKL, q-norm MKL, multiclass MKL) (only Shogun of all)
  * Linear Programming Machine (linear classifier)
  * Markov Chains
  * Barnes-Hut t-SNE (dimension reduction)
  * String Kernels
  * Optimization bindings to CPLEX, Mosek, BFGS
  * Structural output learning (Label Sequence Learning, Factor Graph Learning, SO-SGD, Latent SO-SVM)
  * HDF5 storage
* Sklearn has more than Shogun:
  * Visualization
  * Semi-supervised learning
  * Image processing
  * Decision trees
  * feature selection (Forward, Recursive Feature Selection)
* Shogun has more than Weka:
  * Structured Output Learning
  * Large Scale Learning
  * Multitask Learning
  * Domain Adaptation
  * many language bindings
  * SVM solvers
  * Kernel Ridge Regression
  * multiple kernel learning
  * LASSO, Linear Programming Machine, LDA
  * Markov Chains, Hidden Markov Models
  * many dimensionality reduction
  * Structural Output Learning
* Weka has more than Sklearn:
  * GUI, C++, Java
  * SVMLight, SVMPegasos (solver)
  * Relevance vector machine (regression)
  * Bayesian Networks, Multi Layer Perceptron, RBF Networks (classifier)
  * String Kernels
  * Wrapper methods (feature selection)
  * Mean value imputation, EM-based/model based imputation
  * BFGS optimization
  * Arff file format
* Features: Shogun 84, Weka 54, Sklearn 54, dlib 42, Orange 35, PyML 31, MLPY 32
* Special:
  * SVMPegasos only in Weka, dlib
  * Relevance Vector machines (Regression) only in Weka, dlib
  * Bayesian Networks, Multi Layer Perceptron, RBF Networks (Classifier) only in Weka, dlib
  * Sammon mapping (dimension reduction) only in dlib, orange
  * feature selection wrapper method only in Weka
  * Mean value imputation, EM-based/model based imputation only in Weka, Orange
  * conjugate gradient optimization only in dlib (of all)
  * ARFF file format only Weka
* Not compared: RapidMiner, KNIME, JHepWork

KNIME:
* based on Eclipse
* -> plugins possible

JHepWork:
* based on Jython
* 2D/3D plots

Rapidminer:
* a lot of ML operations
* can be integrated

MLib:
* on Apache Spark
* for large scale
* standard component of Spark

Statsmodel
..........
* statistical models
* many stats tests
* accepts Numpy and DataFrames

PyMC
....
* best for Bayesians
* hierarchical models

Shogun
......
* focus on SVM
* Python interface
* great speed

Gensim
......
* topic modeling for humans
* main focus Latent Dirichlet Allocation
* NLP support
* Recurrent Neural Network text representation called word2vec
* pure Python

Theano
......
* most mature deep learning

PyLearn2
........
* on top of Theano
* neural networks by file configuration
* more modularity

Decaf
.....
* state of the art NN by Berkeley

Nolearn
.......
* wraps Decaf for scikit

OverFeat
........
* won Kaggle
* GPU by Torch
* mainly computer vision

Hebel
.....
* NN with GPU support
* config my YAML
* new (little docs)
* limited in NN models
* pure Python
* adds schedulers and monitors

Neurolab
........
* different variantes of Recurrent Neural Networks
* simple API

Inactive (1year no update): MDP, MIPy, FFnet, PyBrain

Workflow
--------
Apache Oozie
............
* Java Web application to schedule Hadoop jobs
* combines multiple jobs sequentially
* supports MapReduce, Pig, Hive, Sqoop

Drake
.....
* https://github.com/Factual/drake
* "make for data"
* text-based data workflow
* organizes commands and deps
* HDFS support
* multiple inputs/outputs

Chronos
.......
* http://airbnb.github.io/chronos/
* by AirBnB
* raw bash scheduler written in Scala; flexible, fault-tolerant, distributed
* maintain complex workflows
* web UI for business analysts: define, execute, and monitor workflows: a zoomable DAG highlights failed jobs and displays stats that can be used to identify bottlenecks
* async jobs

Luigi
.....
* https://github.com/spotify/luigi
* Python module by Spotify
* dependency resolution, workflow management, visualization, handling failures, command line integration, ...

Data table analysis
-------------------
Hive
....
* Alternatives:
  * Phoenix on HBase
  * Impala
  * Hive on Tez/Spark
  * PrestoDB
  * Spark SQL

DataWrangler
............
* http://vis.stanford.edu/wrangler/
* recommended research project
* no longer supported, but commerical venture Trifacta instead
* mark lines -> Wrangler automatically suggests operations!
* bit more "business" than OpenRefine
* simple pivots
* generates javascript for clean-up

OpenRefine
..........
* http://openrefine.org/
* Videos: https://www.youtube.com/watch?v=B70J_H_zAWM, https://www.youtube.com/watch?v=5tsyz3ibYzk, https://www.youtube.com/watch?v=5tsyz3ibYzk
* Plugin: https://www.bits.vib.be/index.php/software-overview/openrefine
* pretty powerful
* filter, histogram, rename, transform, script expressions, even parse (custom, JSON, etc), ...

Hue
...
* http://gethue.com/
* Web interface to analyze data with Hadoop

New startups
............
* Paxata
* Trifacta
* DataTamer

BI Startups
...........
* birst
* looker
* insightsquared
* chart.io
* domo
* gooddata
* edgespring
* datapad

Predictive analytics startups
.............................
* Alpine data labs
* alteryx
* skytree
* wise.io
* bigml

Palladium (Otto): https://github.com/ottogroup/palladium
Palladium provides means to easily set up predictive analytics services as web services. It is a pluggable framework for developing real-world machine learning solutions. It provides generic implementations for things commonly needed in machine learning, such as dataset loading, model training with parameter search, a web service, and persistence capabilities, allowing you to concentrate on the core task of developing an accurate machine learning model.

Aerosolve (Airbnb): https://github.com/airbnb/aerosolve
A machine learning library designed from the ground up to be human friendly. It is different from other machine learning libraries in the following ways:
    A thrift based feature representation that enables pairwise ranking loss and single context multiple item representation.
    A feature transform language gives the user a lot of control over the features
    Human friendly debuggable models
    Separate lightweight Java inference code
    Scala code for training
    Simple image content analysis code suitable for ordering or ranking images

Airflow (Airbnb) http://nerds.airbnb.com/airflow/

Most popular ML libraries on github (by contributors) http://www.kdnuggets.com/2015/06/top-20-python-machine-learning-open-source-projects.html:
* sklearn 404
* pylearn2 (117) Pylearn2 is a library designed to make machine learning research easy. Its a library based on Theano
* NuPIC (60) The Numenta Platform for Intelligent Computing (NuPIC) is a machine intelligence platform that implements the HTM learning algorithms. HTM is a detailed computational theory of the neocortex. At the core of HTM are time-based continuous learning algorithms that store and recall spatial and temporal patterns. NuPIC is suited to a variety of problems, particularly anomaly detection and prediction of streaming data sources
* Nilearn (28) Nilearn is a Python module for fast and easy statistical learning on NeuroImaging data. It leverages the scikit-learn Python toolbox for multivariate statistics with applications such as predictive modeling, classification, decoding, or connectivity analysis
* Pybrain (27) PyBrain is short for Python-Based Reinforcement Learning, Artificial Intelligence and Neural Network Library. Its goal is to offer flexible, easy-to-use yet still powerful algorithms for Machine Learning Tasks and a variety of predefined environments to test and compare your algorithms
* Pattern (20) Pattern is a web mining module for Python. It has tools for Data Mining, Natural Language Processing, Network Analysis and Machine Learning. It supports vector space model, clustering, classification using KNN, SVM, Perceptron
* Fuel (12) Fuel provides your machine learning models with the data they need to learn. it has interfaces to common datasets such as MNIST, CIFAR-10 (image datasets), Google's One Billion Words (text). It gives you the ability to iterate over your data in a variety of ways, such as in minibatches with shuffled/sequential examples
* Bob (11) Bob is a free signal-processing and machine learning toolbox The toolbox is written in a mix of Python and C++ and is designed to be both efficient and reduce development time. It is composed of a reasonably large number of packages that implement tools for image, audio & video processing, machine learning and pattern recognition

Tool poll http://www.kdnuggets.com/2015/05/poll-r-rapidminer-python-big-data-spark.html
Good: R, Rapidminer, SQL, Python, Excel, KNIME
Increase: H20, Actian, Spark, MLib, Alteryx, Python, Pig, ...
Decrease: Alpine, Octave, Revolution Analytics
Deep learning: Pylearn2, Theano

Deep learning http://www.kdnuggets.com/2015/06/popular-deep-learning-tools.html
Theano + Pylearn2: Theano and Pylearn2 are both developed at University of Montreal with most developers in the LISA group led by Yoshua Bengio. Theano is a Python library, and you can also consider it as a mathematical expression compiler. It is good for making algorithms from scratch. Here is an intuitive example of Theano training. If we want to use standard algorithms, we can write Pylearn2 plugins as Theano expressions, and Theano will optimize and stabilize the expressions. It includes all things needed for multilayer perceptron/RBM/Stacked Denoting Autoencoder/ConvNets. Here is a quick start tutorial to walk you through some basic ideas on Pylearn2.
Caffe: Caffe is developed by the Berkeley Vision and Learning Center, created by Yangqing Jia and led by Evan Shelhamer. It is a fast and readable implementation of ConvNets in C++. As shown on its official page, Caffe can process over 60M images per day with a single NVIDIA K40 GPU with AlexNet. It can be used like a toolkit for image classification, while not for other deep learning application such as text or speech
Torch + Overfeat: Torch is written in Lua, and used at NYU, Facebook AI lab and Google DeepMind. It claims to provide a MATLAB-like environment for machine learning algorithms. Why did they choose Lua/LuaJIT instead of the more popular Python? They said in Torch7 paper that “Lua is easily to be integrated with C so within a few hours’ work, any C or C++ library can become a Lua library.” With Lua written in pure ANSI C, it can be easily compiled for arbitrary targets. cudnnOverFeat is a feature extractor trained on the ImageNet dataset with Torch7 and also easy to start with.
Cuda: There is no doubt that GPU accelerates deep learning researches these days. News about GPU especially Nvidia Cuda is all over the Internet. Cuda-convnet/CuDNN supports all the mainstream softwares such as Caffe, Torch and Theano and is very easy to enable.
Deeplearning4j: Unlike the above packages, Deeplearning4j is designed to be used in business environments, rather than as a research tool. As on its introduction, DL4J is a “Java-based, industry-focused, commercially supported, distributed deep-learning framework.”

Beaker http://beakernotebook.com/features: multi-language notebook

Application Hadoop:
* Streaming: Simple job
* Luigi: Complex ETL, integration with other processes
* mrjob: AWS
* Impala+Numba: Python and high performance, UDF for SQL
* Pydoop

Data crunching
==============
SFrame
------
* Scalable Data Frame, open-sourced by Dato
* out-of-core analytics

SGraph
------
* out-of-core Graph analytics, open-sourced by Dato

Dask
----
* out-of-core scheduler, execution engine
* pure Python
* divide data into chunks and use parallel computations
* encode data as dask collections

Ibis
----
* mirrors single node experience in Python, by Cloudera

Blaze
-----
* like query optimizer

Splash
------
* Javascript rendering engine for web scraping
* web scraping when heavy Javascript present, in Twisted and Qt
* light-weight web browser with HTTP API

Pentuum
-------
* http://petuum.github.io/
* distributed machine learning
* Bösen: async distributed key-value store
* Strads: dynamic scheduler for model-parallel ML, performs find-grained scheduling

Flink
-----
* scalable batch and stream data processing, by Apache
* wants to replace MapReduce
* does streaming processing (unlike Spark which does mini-batches)

Pyxley
------
* developing web apps/dashboard for Python
* enabled through Flask, PyReact, Pandas

Python libraries
----------------
Modeling
........
* missingno (https://github.com/ResidentMario/missingno): missing rows analysis
* mlxtend (http://rasbt.github.io/mlxtend/): useful additions to data science work with sklearn
* pandas-summary (https://github.com/mouradmourafiq/pandas-summary): more detailed summary stats for pandas
* polyssifier (https://github.com/alvarouc/polyssifier): run multiple classifiers on data; alpha
* sklearn-deap (https://github.com/rsteca/sklearn-deap): evolutionary parameter search
* tpot (https://github.com/rhiever/tpot): genetic pipeline optimization
* sklearn-evaluation (https://github.com/edublancas/sklearn-evaluation): nicer plotting and report of results
* himdim decision boundary plot (https://github.com/tmadl/highdimensional-decision-boundary-plot): visualize classification by high dim onto 2D
* qgrid (https://github.com/quantopian/qgrid): interactive grid for data frames
* slickgrid (https://github.com/6pac/SlickGrid/wiki): powerful js table display
* categorical encoding (https://github.com/wdm0006/categorical_encoding)
* yamal/blue-yonder (not open source yet?): interesting tool for date pipelines
* https://github.com/tmadl/highdimensional-decision-boundary-plot: visualize decision region and explore
* jmpy (https://github.com/beltashazzer/jmpy): some visualization
* https://github.com/jbrambleDC/simulacrum: create simulated data
* feature-selection (http://featureselection.asu.edu/)
* hyperopt-sklearn (http://hyperopt.github.io/hyperopt-sklearn/)
* scikit-plot (https://github.com/reiinakano/scikit-plot)
* supersmoother (https://github.com/jakevdp/supersmoother/)
* scikit-garden (https://github.com/scikit-garden/scikit-garden): Additional forest models
* sympy.stats (http://docs.sympy.org/latest/modules/stats.html): Basic operations with random variables
* PyMC3
* statsmodels
* patsy


ML technique
............
* cluster (https://pypi.python.org/pypi/cluster/): deal with clusterings
* mord (https://pythonhosted.org/mord/): ordinal regression
* lda2vec (https://github.com/cemoody/lda2vec): word2vec + LDA
* pca-magic (https://github.com/allentran/pca-magic): PCA with missing values
* python mapper (http://danifold.net/mapper/introduction.html): topological data analysis
* t-digest (https://github.com/tdunning/t-digest): rank-based statistics
* online multiclass lpboost (https://github.com/amirsaffari/online-multiclass-lpboost)
* rf implementations (http://stats.stackexchange.com/questions/10001/optimized-implementations-of-the-random-forest-algorithm)
* kmc2 (https://pypi.python.org/pypi/kmc2): K-Means seeding
* sklearn-expertsys (https://github.com/tmadl/sklearn-expertsys): rule learner
* autosklearn-zeroconf (https://github.com/paypal/autosklearn-zeroconf)

Data structures
...............
* banyan (https://pypi.python.org/pypi/Banyan): search trees (not updated anymore)
* intervaltree (https://pypi.python.org/pypi/intervaltree/2.1.0): editable interval tree data structure
* lazysorted (https://pypi.python.org/pypi/lazysorted/): sorted on access only, partial sort
* bintrees (https://pypi.python.org/pypi/bintrees/): binary trees
* blist (https://pypi.python.org/pypi/blist): faster than list when modifying large lists
* covertree (https://github.com/patvarilly/CoverTree): replacement for sklearn.kdtree
* sortedcontainers (https://pypi.python.org/pypi/sortedcontainers): sorted data structures
* simulatcrum (https://github.com/jbrambleDC/simulacrum): generate simulated data
* vaex (http://vaex.astro.rug.nl/): large data tables; https://www.youtube.com/watch?v=bP-JBbjwLM8; billion rows/sec

Other
.....
* pendulum: recommended for time if not time-critical (see Python Video Notes); [Arrow second best]
* timy (https://github.com/ramonsaraiva/timy): time Python functions
* datashader
* holoviews
* re2 (https://pypi.python.org/pypi/re2/): interface to Google regex; faster, but less functions
* regex (https://pypi.python.org/pypi/regex/): regex with more features
* regexdict (https://github.com/perimosocordiae/regexdict): dict to query keys by substring
* commonregex (https://github.com/madisonmay/CommonRegex): common regex (email, ...)
* castra/blaze (https://github.com/blaze/castra) :  partitioned storage system based on blosc
* snakebite/spotify (https://github.com/spotify/snakebite): Python HDFS client
* etetoolkit (http://etetoolkit.org/): show tree structures
* validator (https://github.com/mansam/validator.py): simple data validation; not updated
* fake-factory (https://pypi.python.org/pypi/fake-factory): generate fake data (e.g. names, ...)
* datadiff (https://pypi.python.org/pypi/datadiff): diffs between Python data structures
* ftfy (http://ftfy.readthedocs.io/en/latest/): fix unicode
* textract (http://textract.readthedocs.io/en/latest/): extract data from many file formats (PDF, ...)
* xlwings (http://xlwings.org/): Excel
* simhash (https://github.com/seomoz/simhash-py): near duplicate detection
* xxhash (https://github.com/Cyan4973/xxHash): hast hash
* bcolz (http://bcolz.blosc.org/): compressed columnar storage
* xarray (http://xarray.pydata.org/en/stable/): N-dim array; formerly xray; replaces pandas Panels
* lifter (https://github.com/EliotBerriot/lifter): query iterables
* booby (https://pypi.python.org/pypi/booby): data modeling and validation; not updated anymore
* bubbles (http://bubbles.databrewery.org/index.html): data processing and quality measurement
* data-tools (https://github.com/clarkgrubb/data-tools): format conversion, sampling, light editing

* FALCONN (https://pypi.python.org/pypi/FALCONN): similarity search on high-dim data with LSH
* feature-forge (http://feature-forge.readthedocs.io/en/latest/feature_definition.html): feature experimenting
* fuel (https://github.com/mila-udem/fuel): feed models with data (e.g. chunking)
* hiscore (https://github.com/aothman/hiscore): create scoring functions
* pyculiarity (https://github.com/nicolasmiller/pyculiarity): twitters anomaly detection in Python
* semisup-learn (https://github.com/tmadl/semisup-learn): semi-supervised learning
* engarde (https://github.com/TomAugspurger/engarde): basic data validation
* changefinder (https://pypi.python.org/pypi/changefinder): change-point detection (not updated)
* crossfader (https://github.com/better/crossfader): find structure in any dataset
* palladium/otto (https://github.com/ottogroup/palladium): setting up ML services
* pycast (https://github.com/T-002/pycast): forecasting and smoothing
* lea (https://bitbucket.org/piedenis/lea): working with discrete probability distributions
* probpy (https://github.com/petermlm/ProbPy): probabilistic calculus
* augustus (https://code.google.com/archive/p/augustus/): PMML model consumer; larger than memory
* featureimpace (https://pypi.python.org/pypi/featureimpact): test statistical feature impact on sklearn classifier; not updated
* pyensemble (https://github.com/dclambert/pyensemble): Caruana's ensemble selection
* pyfeast (https://github.com/mutantturkey/PyFeast): feature selection; not updated
* panns (https://github.com/ryanrhymes/panns): approx nearest neighbor search
* platypus (http://platypus.readthedocs.io/en/latest/): multi-objective optimization
* recipy (https://github.com/recipy/recipy): data provenance
* dedupe (https://github.com/datamade/dedupe): de-duplication
* joblib (https://pythonhosted.org/joblib/): cached pipelining
* pydatalog (https://sites.google.com/site/pydatalog/): logic programming
* pyrind (https://github.com/rasbt/pyprind): progress indicator, with memory stats
* slugiy (https://github.com/un33k/python-slugify): remove unicode and special characters
* jellyfish (https://github.com/jamesturk/jellyfish): approximate string match
* sumatra (http://neuralensemble.org/sumatra/): data provenance
* pretex (https://github.com/s9w/preTeX): latex simplified syntax processor
* pyxll (https://www.pyxll.com/): python in excel
* datanitro (https://datanitro.com/): excel automatization
* palettable (https://jiffyclub.github.io/palettable/): color palettes
* ipython_memory_usage (https://github.com/ianozsvald/ipython_memory_usage): show deltas
* gooey (https://pypi.python.org/pypi/Gooey/): command line into GUI
* pyhive (https://github.com/dropbox/PyHive): Hive access
* rftk (https://github.com/david-matheson/rftk): random forest toolkit; not updated?
* pivottable_js (https://github.com/nicolaskruchten/jupyter_pivottablejs): interactive pivot table
* pytree (https://github.com/yoyzhou/pyTree): simple tree structure and display; not updated
* treecompare (https://github.com/rubyruy/treecompare): compare tree structures
* plotbrowser (https://github.com/allthedata/plotbrowser): change plot appearance by dialog
* grasp (https://pypi.python.org/pypi/grasp): inspect object data and structure
* DyND more flexibly array type (solves Numpy problems)

Visualization
-------------
* http://glueviz.org/en/stable/: extensible!
* https://orange.biolab.si/: some useful interactive plots
* https://github.com/nicolaskruchten/jupyter_pivottablejs: multiple plots right in Jupyter
* https://pair-code.github.io/facets/: Simple TableProfiler like visualization and basic interactive scatter plot
* https://github.com/cmpolis/datacomb: sort, filter, scatter, histogram
* https://github.com/jwkvam/bowtie: create dashboards with Python

Other
-----
* regex generator online (http://regex.inginf.units.it/index.html)
* data tools (https://github.com/clarkgrubb/data-tools): small data tools (e.g. file conversion, reservoir sampling, ...)
* harry (http://www.mlsec.org/harry/): string similarity

Dead
----
* Blaze, Castra, Odo(?): originally could run on multiple engines
*

scikit-optimize
---------------
* tree based: if high dim and/or param of diff type/scaling
* tree since any sklearn estimator if predict(return_std=True)
* dummy_minimize: random
* skopt.Optimizer: "ask" next wanted values; "tell" function value
