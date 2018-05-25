Data Mining
===========

Process
-------
add missing data
drop outliers

Kaggle learnings
----------------
* don't overfit
* not needed: all predictors, all data rows
* tune algorithm
* reduce data set (average, select particular samples)
* explore visually
* submit simple solutions too
* keep history of submissions


= Rules =
* For Every Result, Keep Track of How It Was Produced
* Avoid Manual Data Manipulation Steps
* Archive the Exact Versions of All External Programs Used
* Version Control All Custom Scripts
* Record All Intermediate Results, When Possible in Standardized Formats
* For Analyses That Include Randomness, Note Underlying Random Seeds
* Always Store Raw Data behind Plots
* Generate Hierarchical Analysis Output, Allowing Layers of Increasing Detail to Be Inspected
* Connect Textual Statements to Underlying Results
* Provide Public Access to Scripts, Runs, and Results

= Tom Khabazaâ€™s 9 Laws of Data Mining=
* Business objectives are the origin of every data mining solution
* Business Knowledge is central to every step of the data mining process (business knowledge required)
* Data preparation is more than half of every data mining process
* The right model for a given application can only be discovered by experiment
* There are always patterns (Watkins' Law)
* Accuracy or stability in a model are good, of course, but may be less important than issues such as the importance of predicted values to a business, meaningful insights, or the ease of putting the predictions to use

= Eslami - Patterns for Research in Machine Learning [[http://arkitus.com/PRML/ Link]] =
* Use version control (to collaborate, replicate, work from multiple computers)
* Separate code from data (/data, /code; to share code, swap data; hide data from version control)
* Separate input/working/output data (input: never change; working: always change, can safely be deleted; output: for sharing)
* Always keep data raw (keep note where from and licensing; write script to convert data; never clean by hand, or document thoroughly)
* Safe to disk frequently; store output of different days to different folders
* Separate Options (how algorithm should run) and Parameters (fitted model, output of algorithm); unaffected parameters can be initialized by options
* Don't use global variables; communicate through function arguments (easier to debug and parallize)
* Record options used at each run (set random number seed; save copy of code for important runs)
* Make it easy to sweep options (ranges returned by ``get_config``)
* Make it easy to execute only portions of code (specify which parts; store intermediate results, ``run_experiment('dataset_1_options', '|preprocess_data|initialise_model|train_model|')``)
* Use checkpoints; experiments fail occationally; store entire state (counters etc.) at suitable intervals; code should be able to continue from latest saved state (write message that it restarts)
* Write demos and tests (/code, /data, /demos, /tests)
* Estimate how long experiment will run
* Keep journal that explains why you ran experiment and what finding are
* Do coding and debugging on subset of data that runs in 10sec
* Make it easy to swap in and out different models
* Record Git revision number of code
* Use GNU make for pipeline? (download, traning, evaluating)

= Other =
* understand what to optimize
* to be good: model features; to win: perfect algorithm optimization
* data files read-only


= Consider =
* speed (training, scoring)
* interpretability
* simplicity (little tuning for production and maintainance)
* scalability
* visualize
* try RF, SVM, elastic net
* improve: blending, individual models, feature engineering, translation of solution to optimal submission

= Organisation =
* HDF for data
* IPython notebooks or Pweave reports
* state all parameters in filename
* ideas: discrete tasks; tasks store results; tasks accept own arguments and have own dependencies; tasks have caching and storage; tasks testable

= Data overview =
* common sense for features
* read forums
* visualize
* look at logreg feature weights
* visualize decision tree
* cluster data and observe
* look at raw data and labels
* train simple classifier and see where mistakes
* write classifier with handwritten rules
* use fancy method
* read literature
* use ensembles
* human decision on uncertain predictions
* make many features and chose my L1 sparsity
* monitor: accuracy on test data, input feature distribution, output score distribution, output classification distribution

Large data techniques
* dimension reduction:
  * sklearn.random_projection
  * sklearn.cluster.WardAgglomeration
  * sklearn.feature_extraction.text.HashingVectorizer
* online algorithms
  * sklearn.MiniBatchKMeans
* parallel processing
* caching:
  * joblib.Memory
* fast IO:
  * zlib.compress (avoid copies: zlib needs C-contiguous buffers; store raw buffer and meta info; use __reduce__; rebuild np.core.multiarry._reconstruct)
  * pytables (even faster than joblib)
* joblib for pipeline-ish patterns
