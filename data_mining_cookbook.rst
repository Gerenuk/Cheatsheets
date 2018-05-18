Loading data
============
http://pandas.pydata.org/pandas-docs/version/0.13.1/generated/pandas.DataFrame.from_csv.html


Scaling
=======
..highlight::

	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	
kNN
===
Use custom metric with::
    
    def mydist(x, y):
        return np.sum((x-y)**2)

    nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree', metric='pyfunc', func=mydist)
    
Encoding
========
Categorical variables
::
    df_str=map(itemgetter(1), df[features].iterrows()) # only string values will be coded
    dv = DictVectorizer()
    X = dv.fit_transform(df_str).toarray()  # toarray needed for some classifiers
    
Storing results
===============
For storing large data to disk, `sklearn.joblib` can make smaller files (http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence):
::
    from sklearn.externals import joblib
    joblib.dump(clf, 'filename.pkl')
    clf=lobjob.load('filename.pkl')
    
Testing
=======
::
    from sklearn.cross_validation import train_test_split
    X, X_test, y, y_test = train_test_split(data, target, test_size=.33, random_state=1111)
    
    parameters = [{'alpha':np.linspace(0.1,1,10)}]
    from sklearn.grid_search import GridSearchCV
    clf = GridSearchCV(CLF(), parameters, cv=10, scoring='f1')
    clf.fit(X,y)
    
    res = zip(*[(f1m, f1s.std(), p['alpha']) for p, f1m, f1s in clf.grid_scores_]) # get score and stddev
    
    best_alpha = clf.best_params_['alpha']
    from sklearn.metrics import f1_score
    final = f1_score(y_test, clf.best_estimator_.predict(X_test))

    
Analyze results
===============
Visualize decision tree:
::
    from sklearn import tree
    from io import StringIO
    for num in range(10):
        dot_data = StringIO() 
        tree.export_graphviz(clf.estimators_[num], out_file=dot_data, max_depth=2, feature_names=clf.feature_names)
        %dotstr dot_data.getvalue()
