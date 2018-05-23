Machine learning
================

Notable techniques
------------------

Asynchronous Distributed Gradient Descent: big data
L-BFGS: big data


          features=["...", ...]
          d=df.dropna(subset=features)
          X=np.array([d[f] for f in features]).T
          y=np.array(d.CLASS)
          
          # Random forest classifier
          from sklearn.ensemble import RandomForestClassifier
          clf=RandomForestClassifier(n_estimators=20, oob_score=True, compute_importances=True)
          clf=clf.fit(X,y)
          
          # Cross validation
          from sklearn import cross_validation
          print(cross_validation.cross_val_score(clf, X, y, cv=5))
          
          
scikit-learn.org/0.13/modules/generated/sklearn.metrics.matthews_corrcoef.html
sklearn.metrics.matthews_corrcoef(y_true, y_pred)

http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
sklearn.preprocessing.normalize(X, copy=False)

Notes:
* Decision tree metrics:
  * Gini more for continuous, Entropy for classes (http://paginas.fe.up.pt/~ec/files_1011/week%2008%20-%20Decision%20Trees.pdf)
  * Gini tends to find largest class, Entropy groups that make ~50%
  * Gini to minimize misclassification, Entropy for exploratory analysis
  * usually Gini and Entropy the same within 2% (https://rapid-i.com/rapidforum/index.php?topic=3060.0)
  * Entropy slightly slower to compute
  

PRIM: http://ir.library.louisville.edu/cgi/viewcontent.cgi?article=2454&context=etd
