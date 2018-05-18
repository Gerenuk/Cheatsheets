Pandas cookbook
===============

Missing values
--------------
General information http://pandas.pydata.org/pandas-docs/dev/missing_data.html

Drop missing values
...................
Use http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html

    df.dropna(subset=["col1", "col2"])
    
or

    df=df[pd.notnull(df.col1) & pd.notnull(df.col2)]
    
http://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
np.unique() find unique elements on array (returns sorted)
can also return the indices

Configuration
-------------
Call ``pd.describe_option()`` to list all options (http://pandas.pydata.org/pandas-docs/stable/basics.html#working-with-package-options).
::
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
Selecting data
--------------
Make sure to use parenthesis ``(df.a == 1) & (df.b == 1)`` or you get ``The truth value of a Series is ambiguous``.
::
    df[df.col.map(lambda x:<cond>)]   # select by function
    df.ix[df.groupby("GROUP").VAR.idxmax()]   # select row where VAR is max within GROUP
    
Write data
----------
::
    df["new"]=df.A.map(str)+" "+df.B
    df["a"], df["b"] = zip(*df.map(func_return_tuples))  # single col to multi col
    df["a"] = df.apply(func_on_series, axis=1)  # multi col to single col
    df.apply(func_return_series, axis=1)  # multi col to multi col
    df.ix[df.groupby("GROUP").COL.idxmax(),["COL"]]=True
    df.ix[CONDITION, ["COL"]]=DATA
    
Index
-----
::
    df.reset_index()   # make all indices to normal data columns
    
Information
-----------
::
    df.describe()
    df.cov()
    df.corr()
    
Process data
------------
::
    df.apply(lambda x:x["a"]+x["b"], axis=1)
    
Vectorized string operations: http://pandas.pydata.org/pandas-docs/stable/basics.html#vectorized-string-methods
    
Aggregating
-----------
::
    df.groupby("key").myvar.agg(myfunc)
    df[...].groupby(level=0)    # groupby index
    
Columns
-------
::
    df.rename(columns={"old":"new"})
    
Tips
----
For sklearn us ``df.values.astype(float32)`` to avoid ``object dtype``.

pandas.DatetimeIndex.asof¶


Taking Series from DF fastest
df.blocks -> dict dtype:numpy
caching for getting column values

instead of df[[cols...]]
idxs = df.columns.get_indexer([cols...])  # could precalculate this
df.take(idxs, axis=1)

.iloc diff from [] (inclusive?)

df[col].values[i:j]  faster since direct numpy

index.get_loc(..)

df.loc[]=..

most functions trigger block consolidation


