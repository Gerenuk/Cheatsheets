Supervised learning
+++++++++++++++++++

Linear model
============
* ``linear_model.LinearRegression()``
* relies on independence; when features correlated, matrix becomes close to singular and this highly sensitive to noise
* ``linear_model.Ridge()``
* ``linear_model.Lasso()`` (uses coordinate descent; note that :math:`\alpha` is different scale than Ridge due to factor :math:`1/(2n)`)
* ``linear_model.LassoCV()`` (preferable for high-dimensional with many collinear regressors)
* ``linear_model.LassoLarsCV()`` (based on least angle regression; explores more :math:`\alpha` parameters than LassoCV; for few samples faster than LassoCV)
* ``linear_model.LassoLarsIC()`` (uses AIC or BIC; computationally cheaper; assumes data is really this model; needs proper estimation of degree; breaks down when badly conditioned [more features than samples])
* ``linear_model.ElasticNet()`` (useful when multiple features which are correlated [unlike Lasso L1 which will pick only one of the features]; inherits stability of Ridge under rotation)
* ``linear_model.ElasticNetCV()`` (set parameters by CV)
* ``linear_model.MultiTaskLasso()`` (uses same sparse coefficients for multiple problems)
* ``OrthogonalMatchingPursuit()`` (L0 norm)
* ``Lars`` algorithm (numerically efficient if :math:`p\gge n`; full piecewise linear solution path [useful for CV and tuning]; equally correlated variables get similar coefficients; can be sensitive to noise due to iterative refitting of residuals)
* complexity: for a matrix X (:math:`n \times m`) -> :math:`O(nm^2)`
* plot Ridge coefficients (`sklearn <http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html#example-linear-model-plot-ridge-path-py>`_)
* built-in cross-validation of :math:`\alpha`: ``linear_model.RidgeCV()`` (like ``GridSearchCV`` but defaults to an efficient form of leave-one-out)
* Bayesian Regression: weights distributed with precision :math:`\lambda`; :math:`\lambda` and regularization parameter :math:`\alpha` estimated from data (without CV); adapts to data; can be slow
* ``BayesianRidge()`` (weights are spherical Gaussian; use gamma distribution priors for regularization parameter and weight width [other than standard Gaussian])
* ``ARDRegression`` (Automatic relevance determination; weight distribution not spherical but axis-parallel; leads to sparser weights)
* ``LogisticRegression()`` (for L1 you can use ``sklearn.svm.l1_min_c`` to estimate lower bound for :math:`C` to get non null model)
* ``SGDClassifier``, ``SGDRegressor``: useful when very many samples
* ``Perceptron()``: does not need learning rate; not regularized; updates only on mistakes (therefore slightly faster then SGD and result sparser)
* ``PassiveAggressiveClassifier()``, ``PassiveAggressiveRegressor()``: for large-scale learning; similar to Perceptron but dont need learning rate; are regularized with parameter :math:`C`

Support Vector Machines
=======================
* high effective in high-dim space (even when more dimensions than samples)
* uses subset of training points -> memory efficient
* but poor performance if much more features than samples
* no direct probability estimates (internally 5-CV used)
* supports numpy.ndarray and scipy.sparse (however same for fit and predict)
* fastest with numpy.ndarray or scipy.sparse.csr_matrix dtype=float64
* ``SVC``, ``NuSVC``: similar but different parameters
* ``LinearSVC``: other implementation
* ``clf.n_support_``: number of support vectors
* multiclass:
  * ``SVC``: one-vs-one multiclass (therefore :math:`n_c(n_c-1)/2` classifiers)
  * ``LinearSCVC``: one-vs-rest (prefered); or ``multi_class='crammer_singer'`` possible (consistent method but much slower)
* with ``probability=True`` Platt scaling is used (LogReg on SVM scores fit by CV); very slow; also some theorical issues with Platt; preferably use ``decision_function`` for confidence scores
* ``SVC`` only: parameter ``class_weight``; sets parameter to :math:`C\cdot\mathrm{weight}`
* all (but LinearSVC) have ``sample_weight``
* ``SVR``, ``NuSVR``: support vector regression
* ``OneClassSVM``: find soft (cluster) boundary; for novelty detection
* complexity:
  * :math:`O(n_\mathrm{features}n_\mathrm{samples}^{2\ldots 3})` (use average number of features if sparse)
  * ``LinearSVC``: much more efficient, scales almost linearly (millions of samples and/or features)
* tips:
  * check if input data contiguous (inspect flag attributes); otherwise will be copied; however ``LinearSVC`` will always copy (use ``SGDClassifier`` to avoid that)
  * for large problems and enough RAM increase ``cache_size`` to higher value
  * decrease :math:`C` for noisy observations
  * parameter ``nu`` approximates fraction of training errors and support vectors
  * ``LinearSVC`` uses some random numbers; decrease ``tol`` to avoid effect
  * ``LinearSVC(loss='l2', penalty='l1', dual=False)`` yields sparse solution
  
Stochastic Gradient Descent
===========================
* easily problems with 10,000 features and 10,000 samples
* needs number of iterations
* sensitive to feature scaling
* adviced to use ``shuffle=True``
* loss functions:
  * ``loss='hinge'``: soft-margine linear SVM (lazy: update only on violation of margin -> sparser model; this one has no ``predict_proba``)
  * ``loss='modified_huber'``: smoothed hinge loss (lazy); less steep than squared hinge
  * ``loss='log'``: logistic regression
* regression with ``SGDRegressor``: useful for >10,000 samples (use conventional regression for smaller problems); loss functions:
  * ``loss='square_loss'``: ordinary
  * ``loss='huber'``: robust regression
  * ``loss='epsilon_insensitve'``: linear SVR; need insensitive region width parameter ``epsilon``
* sparse implementation gives slightly different results due to a shrunk learnign rate for intercept
* use ``scipy.sparse.csr_matrix`` for best performance with sparse
* complexity: linear in samples, iterations and features (runtime for given accuracy does not increase with larger training sizes)
* sensitive to features scaling
* recommended: ``alpha`` by grid search with ``10.0**-np.arange(1,7)``; ``n_iter = np.ceil(10**6 / n)`` since usually convergence after 1,000,000 samples
* if SGD on PCA transform: scale by constant such that average L2 norm is 1

Nearest Neighbors
=================
* ``sklearn.neighbors``
* methods:
  * by k-nearest ``KNeighborsClassifier``
  * fixed radius ``RadiusNeighborsClassifier`` (here, if identical distance, then depends on order in training); better if data not uniformly sampled
* non-generalizing ML methods since only remember points
* successful in some image tasks
* numpy arrays (many distance metrics) or scipy.sparse (arbitrary Minkowski distance)
* ``NearestNeighbors`` (wrapper for ``KDTree`` and ``BallTree``):
  * ``algorithm='auto'``: determined (currently ``'ball_tree'`` if :math:``k<N/2`` and ``brute`` otherwise)
  * ``algorithm='ball_tree'``: ``BallTree`` (better; nested hyperspheres; construction more costly)
  * ``algorithm='kd_tree'``: ``KDTree`` (generalization of 2D quadtrees; good for low-dimensions <20)
  * ``algorithm='brute'``: based on ``sklearn.metrics.pairwise``
.. highlight::
   nnbrs=NearestNeighbors(...).fit(X)
   distances, indices = nnbrs.kneighbors(X)  # use distances or indices
   nnbrs.kneighbors_graph(X).toarray()       # sparse graph; nearby index are nearby in parameter space -> approx block-diagonal
* for metrics see `DistanceMetric <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric>`_
* ``distance=...`` for weighting (uniform and majority vote otherwise)
* regression with mean assigned:
  * ``KNeighborsRegressor``
  * ``RadiusNeighborsRegressor``
* ``leave_size`` parameter: when to switch to brute force
* ``NearestCentroid`` classifier:
  * represents each class by centroid
  * no parameters (good baseline)
  * non-convex
  * assumes equal variances (better: ``sklearn.lda.LDA`` or ``sklearn.qda.QDA``)
  * ``shrink_threshold`` parameter: normalize feature distances, reduce by parameter and cut below zero -> noisy data won't affect classification
  
Gaussian Processes
==================
* usually regression
* here also post-processing for classification possible
* adv:
  * interpolates observations
  * predictions probabilistic -> compute empirical confidence intervals
  * linear regression models and correlation models can be specified
* disadv:
  * not sparse (uses all samples)
  * bad in high dimensions (when >30; slow and bad prediction)
  * classification only a post-processing addon
* ``sklearn.gaussian_process.GaussianProcess()``
* ``nugget`` parameter to specify noise for points: for regularization (adding to diagonal) [if squared-exponential correlation this is equivalent to fractional variance :math:`(\sigma_i/y_i)^2`]
* with correct ``nugget`` and ``corr`` the recovery is robust
* Maths: http://scikit-learn.org/stable/modules/gaussian_process.html#mathematical-formulation
* correlation models:
  * need to know properties of original experiment
  * often matches SVM kernels
  * if infinitely differentiable (smooth): use squared-exponential loss
  * use exponential correlation model otherwise
* implementation based on DACE Matlab toolbox

Cross decomposition
===================
* Partial Least Squares (PLS); Canonical Correlation Analysis (CCA)
* find linear relationship between two multivariate datasets (X and Y are 2D arrays)
* latent variable approaches to model covariance between spaces
* " try to find the multidimensional direction in the X space that explains the maximum multidimensional variance direction in the Y space"
* "particularly suited when the matrix of predictors has more variables than observations, and when there is multicollinearity among X values. By contrast, standard regression will fail in these cases"
* ``PLSRegression``, ``PLSCanonical``, ``CCA``, ``PLSSVD``

Naive Bayes
===========
* due to decoupling each distribution can be independently estimated -> helps vs curse of dimensionality
* ``predict_proba`` is a bad estimator here
* ``GaussianNB``: likelihood Gaussian :math:`P(x_i|y)=N(x_i;\mu_y; \sigma_y)`
* ``MultinomialNB``: `http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes`_
* ``BernoulliNB``: `http://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes`_
* suitable for large scale with parameter ``partial_fit`` that can be used incrementally (use data chunks as large a possible for speed)

Decision Trees
==============
* can be visualized
* little data preparation
* numerical and categorical data
* can do multi-output problems
* white-box: can be translated to boolean logic
* possible to validate model with statistical tests
* can easily overfit
* unstable: small variations in data might result in completely different tree
* heuristics don't always yield best tree
* biased tree if some classes dominate
* ``DecisionTreeClassifier``: can be binary [-1, 1] or multiclass [0, ..., K-1]
* ``sklearn.tree.export_graphviz(clf, out_file=...)``
* with pydot
>>> from sklearn.externals.six import StringIO  
>>> import pydot 
>>> dot_data = StringIO.StringIO() 
>>> tree.export_graphviz(clf, out_file=dot_data) 
>>> graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
>>> graph.write_pdf("iris.pdf") 
* ``DecisionTreeRegressor``
* multi-output:
  * several outputs to predict
  * build single model better if correlations
  * compute average reduction across all outputs
* complexity:
  * construction :math:`O(n_\mathrm{samples}\log(n_\mathrm{samples})n_\mathrm{features})`
  * query time :math:`O(\log n_\mathrm{samples})`
  * scikit learn optimizes by presorting
* tips:
  * overfit when many features (need right ratio of samples to features)
  * use dimensionality reduction before
  * visualize trees
  * requires number of samples doubles for each new tree level
  * try ``max_depth=3`` and `` min_samples_leaf=5`` first
  * balance tree before
  * copy made if not ``numpy.float32``
* CART used here (unlike C4.5 which translates to if-then rules and prunes additionally)

Ensemble methods
================
* Families:
  * averaging (build models independently)
  * boosting (build models sequentially)
* ``RandomForestClassifier``, ``RandomForestRegressor``:
  * bias increased, but variance decreased more
  * sklearn averages probabilistic predictions (instead of single class votes)
* ``ExtraTreesClassifier``, ``ExtraTreesRegressor``:
  * extremly randomized trees
  * also subset of features used at splits; but feature thresholds taken randomly instead of optimally; still best feature selected
  * faster than random forest; sometimes generalizes better than random forest
* ``max_features``:
  * subset of features used at split
  * for regression ``max_features=n_features``
  * for classification ``max_features=sqrt(n_features)``
* ``n_estimators``: more is better, but usually slower; also some threshold
* ``max_depth``: usually ``None`` best with ```min_smaples_split=1``
* feature importance:
  * relative rank in tree (expected fraction of the samples the contribute; top in tree is better)
  * ``feature_importances_`` sum to 1
* ``RandomTreesEmbedding``:
  * unsupervised transformation of data
  * encodes data by indices of leaves the data point ends up in; encoded in one-of-K manner -> high dimensional sparse binary coding
  * size of coding at most :math:`n_\mathrm{estimators}2^{\mathrm{max_depth}}`
* AdaBoost:
  * repeated modified version of data for weak learners; weighted majority vote
  * changing point weights; higher weights for missclassified
  * ``AdaBoostClassifier``: Adaboost-SAMME and Adaboost-SAMME.R
  * ``AdaBoostRegressor``: Adaboost.R2
  * main parameters to tune: ``n_estimators``, complexity of base (``max_depth``, ``min_samples_leaf``)
* Gradient tree boosting (GBRT):
  * generalization of boosting to arbitrary differentiable loss functions
  * can handles mixed data
  * good predictive power
  * robust to outliers
  * hard to parallelize
  * ``GradientBoostingClassifier``: binary and multi-class via deviance loss function (negative binomial log-likelihood)
  * ``learning_rate`` controls overvitting via shinkage (small usually better, e.g. <=0.1; interacts strongly with ``n_estimators`` -> chose by early stopping)
  * ``subsample`` for bagging: subsample alone not good, but with shrinkage often an improvement
  * multi-class needs a tree per class at each iteration -> rather random forest when many classes
  * ``GradientBoostingRegressor`` different loss functions:
    * least squares ``ls``
    * least absolute deviation for robust ```lad``
    * Huber loss ``huber`` which combines least squares and least absolute deviation; parameter ``alpha`` to control sensitivity to outliers
    * quantil loss with parameter ``alpha`` (can be used to create prediction intervals)
  * train error at iterations ``.train_score_``; test error at iterations ``.staged_predict()`` -> determine ``n_estimators`` for early stopping
  * has ``feature_importances_``
  * some initial model used (``init`` argument)
  * solve iterative models by steepest descent
  * ``oob_improvement_[]`` for OOB test estimates (usually pessimistic estimates; use CV is enough time)
* partial dependence plots with ``sklearn.ensemble.partial_dependence.plot_partial_dependence()``:
  * marginalize out all but one or two features
  * for multi-class also select the specific class
  * here for decision trees:
    * if node involves target feature -> follow correct branch
    * otherwise follow both branches
    * in the end average weighted by fraction of samples -> weighted average of all visited leaves
    
Multiclass and multilabel algorithms
====================================
* ``sklearn.multiclass``
* meta-estimators that turn binary or regressor into multiclass
* multiclass: one label each
* multilabel: multiple labels allowed per sample
* multioutput-multiclass: handle jointly several classification tasks; 2D array for y
* useful only if experiment with multiclass strategies
* inherent multiclass: NB, LDA, DT, RF, NN
* multioutput: DT, RF, NN
* ``OutputCodeClassifier``: output code classifier

Feature selection
=================
* transform methods for univariate feature selection:
  * ``SelectKBest``
  * ``SelectPercentile``
  * ``SelectFpr`` false positive rate, ``SelectFdr`` false discovery rate, ```SelectFwe`` family-wise error
  * take input scoring function; return univariate p-values
  * regression: ``f_regression``
  * classification: ``chi2`` (only this useful for sparse data), ``f_classif``
* recursive feature elimination:
  * ``RFE``
  * ``RFECV``: with cross validation
* you can use ``Lasso``, ``LogisticRegression`` or ``LinearSVC`` with L1 norm
* ``LassoCV`` and ``LassoLarsCV`` tends to include too many features; ``LassoLarsIC`` too few
* randomized sparse models:
  * usually sparse model select only one of multiple correlated features
  * -> randomization (perturb design matrix, sub-sampling,...)
  * ``RandomizedLasso``, ``RandomizedLogisticRegression``
  * to be better than standard F statistics at detecting non-zero features, the ground truth should be sparse (there should be only a small fraction of non zero)
* can use tree feature importances

Semisupervised
==============
* ``sklearn.semi_supervised``
* use label ``-1``
* label propagation:
  * classification and regression
  * kernel: rbf (dense matrix, can be slow), knn (more sparse)
  * construct similarity graph
  * ``LabelPropagation``: uses data graph without modification 
  * ``LabelSpreading``: regularized loss, more robust to noise; iterates over modified version of graph (spectral clustering)
  
Linear and quadratic discriminat analysis
=========================================
http://scikit-learn.org/stable/modules/lda_qda.html

Isotonic regression
===================
* tries to find monotic regression (e.g. strictly increasing) curve fitting

Unsupervised learning
+++++++++++++++++++++
http://scikit-learn.org/stable/unsupervised_learning.html

Model selection and evaluation
++++++++++++++++++++++++++++++
Cross validation
================
* ``cross_validation.train_test_split()``
* use "validation" set to tune parameters
* ``cross_validation.cross_val_score()``:
  * different scorings possible
  * can use other strategies by passing cross validation iterator (e.g. ``ShuffleSplit``)
  * when ``cv`` parameter is integer: ``KFold`` (unsupervised), ``StratifiedKFold`` (supervised)
* cross validation iterators:
  * http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
  * boolean masks and indices for split
  * k-fold, stratigied k-fold, leave-one-out, leave-one-label-out, random permutation, bootstrap
  
Grid search
===========
* ``estimator.get_params()`` to get parameters
.. highlight :
   param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  ]
  
* ``scoring`` parameter to change scoring; otherwise ``.score`` used which is ``metrics.accuracy_score`` for classification and ``metrics.r2_score`` for regression
* ``n_jobs=-1`` for parallel
* ``RandomedSearchCV``:
  * budget can be chosen (no wait on irrelevant parameters)
  * ``n_iter`` for iterations
  * can use ``scipy.stats.expon(scale=...)`` etc for sampling (needs ``.rvs()`` method) [seeded by numpy with ``np.random.set_state``
  
Alternatives to brute force search
==================================
* some methods can find parameters as efficient as fitting
* set ``alpha`` as regularizer strength; computer regularization path
* models: ``RidgeCV``, ``RidgeClassifier``, ``LarsCV``, ``LassoLarsCV``, ``LassoCV``, ``ElasticNetCV``
* information criterion: ``LassoLarsIC``
* out-of-bag estimates: ``RandomForest*``, ``ExtraTrees*``, ``GradientBoosting*`` (``*``=``Classifier``,``Regressor``)

Pipeline
========
* chain estimators (only one ``.fit()``)
* grid search of parameters of all estimators at once
* parameter ``[('name', estimator) , ...]``
* call ``.fit()``, ```.transform()`` and pass one to next step
* pipeline has all methods of last estimator
* ``FeatureUnion``:
  * independent fits to transformers
  * sample vectors concatenated end-to-end for larger vectors
  
Model evaluation
================
* scoring parameter can be any callable function (use ``sklearn.metrics.make_scorer`` to fix parameters of given scorers); protocol: called with ``(estimator, X, y)``, return float (higher is better)
* ``sklearn.metric``: ``*_score`` for maximize, ``*_error``/``*_loss`` to minimize
* ``.confusion_matrix(y_true, y_pred)`` computes confusion matrix
* ``print(classification_report(y_true, y_pred, target_names=target_names))``
* for multiclass ``f1_score``, ``fbeta_score``, ``precision_recall_fscore_support``, ``precision_score`` and ``recall_score`` have a ``average`` parameter to specify how to combine scores
* ``matthews_corrcoef(y_true, y_pred)``
* ``fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)``
* also some dummy estimators to simulate random
* ``sklearn.dummy.DummyClassifier(strategy='most_frequent')``: use very simple model

Dataset transformations
=======================
* ``DictVectorizer``: ``[{...},...]`` to numpy array; one-of-K for categorical variables
* ``FeatureHasher``:
  * use hash to determine column index (instead of e.g. a hypothetical super dictionary that maps words to columns)
  * input is ``[{"feat1":val1, ...}, ...]`` or similar; string list input ``["word1", "word2", ...]`` will be converted to ``("word1", 1), ("word2", 1)...``
  * hash sign determines sign of value stored (to cancel effect of possible collisions; use option ``non_negative=True`` if need positive only)
  * prefer ``n_features`` a power of 2
  * result is ``scipy.sparse``
* text feature extraction ``sklearn.feature_extraction.text``:
  * ``CountVectorizer``:
    * transform list (documents) of strings to sparse word count matrix (many parameters for word parsing)
    * methods ``.toarray()``, ``.get_feature_names()``, ``.vocabulary_.get("word")``
  * ``TfidfTransformer()``:
    * reweight to consider frequent words
    * rows normalized (e.g. L2)
    * weights in ``.idf_``
    * ``TfidfVectorizer``: combines ``CountVectorizer`` and ``TfidfTransformer``
    * binary values (option) may be less noisy for short texts
  * ``HashingVectorizer``:
    * no need for memory heavy translation table
    * default ``n_features=2**20``
    * dimensionality does not affect algorithms with CSR matrices (``LinearSVC(dual=True)``, ``Perceptron``, ``SGDClassifier``, ``PassiveAggressive``), but does affect CSC matrices (``LinearSVC(dual=False)``, ``Lasso``, ...)
    * can be used for out-of-core with mini batch (<http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html#example-applications-plot-out-of-core-classification-py>_)
* ``sklearn.features_extraction.image.extract_patches_2d``:
  * extract 2D arrays (or 3D with color)
  * ``image.PatchExtractor``: for pipeline estimator usage or multiple images
* ```img_to_graph`` or ``grid_to_graph`` to encode connectivity between samples

Preprocessing
=============
* ``sklearn.preprocessing``
* ``scale``:
  * scale columns to zero mean and unit variance
  * sparse inputs only if ``with_mean=False``; will be converted to ``sparse.csr_matrix`` (use this upstream)
* ``StandardScaler``: transformer API to be used in pipeline or to reapply transform
* ``MinMaxScaler``:
  * scale to range
  * robust to very small standard deviations
  * preserves zero in entries
* if you need to remove linear correlations: use ``decomposition.PCA`` or ``decomposition.RandomizedPCA`` with ``white=True``
* ``KernelCenterer``: can center kernels
* ``normalize``:
  * normalize each row to unit norm
  * for sparse data use ``scipy.sparse.csr_matrix`` to avoid copies
* ``Normalizer``: tranformer API (but no ``.fit()`` and stateless)
* ``Binarizer``:
  * threshold to boolean
  * no ``.fit()`` method
  * for sparse data use ``scipy.sparse.csr_matrix`` to avoid copies
* ``OneHotEncoder``: one-of-K encoder for categorical features
* ``LabelBinarizer``: utility class to create label indicator matrix from a list of multi-class labels
* ```LabelEncoder``:
  * utility class to rank-normalize labels such that they contain only values ``0`` to ``n_classes-1``
  * can be used on any hashable and comparable (e.g. string)
* ``Imputer``:
  * simple imputation by mean, median, most frequent of row or column, ...
  
Kernel Approxmiation
====================
* explicit (instead of implicit like in SVM) feature mapping -> useful for online learning and reduce cost for very large data sets
* -> use approximate kernel map together with linear SVM
* ``Nystroem``: low-rank approximation of kernels by subsampling of data (default with ``rbf``)
* ``RBFSampler``:
  * approximate (Monte Carlo for ``.fit()``) mapping for rbf kernel
  * less accurate than ``Nystroem`` but faster
* ...

Random projection
=================
* ``sklearn.random_projection``
* approximately preserver pairwise distances (good for distance methods)
* based on Johnson-Lindenstrauss lemma (few points in high dim can be projected to subspace)
* ``johnson_lindenstrauss_min_dim(samples=..., eps=...)``: conservatively estimate minimal subspace size to guarantee some distortion
* ``GaussianRandomProjection``: project to randomly generated matrix where components :math:`N(0,1/n_\mathrm{components})`
* ``SparseRandomProjection``: similar but sparse embedding (memory efficient and faster computation)

Pairwise metrics
================
* evaluate pairwise distances
* modules with distance metrics and kernels (semi-definite)

Example data sets
=================
* ``sklearn.datasets``
* common API
* also random sample generators for many purposes!
* download data from mldata.org

Sparse matrices
===============
* sparse matrices (e.g. from DictVectorizer) are from scipy
* to convert to dense array use ``.toarray()`` (or ``.todense()`` to get ``matrix``)
