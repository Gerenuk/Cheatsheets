Data Structures
===============

General references are

* `Scipy Cookbook <http://wiki.scipy.org/Cookbook/>`_

Input
-----
CSV
...
CSV files can be read with the Python `csv <http://docs.python.org/3/library/csv.html>`_ library
::
    with open("myfile.csv", newline="") as csvfile:
        reader=csv.reader(csvfile, delimiter=" ", quotechar='"') #kwparam dialect for predefined set of parameters (e.g. "excel", "excel-tab")
        for row in reader:
            print(" ".join(row))
            
    with open("myfile.csv", "w", newline="") as csvfile:
        writer=csv.writer(csvfile, delimiter=" ", quotechar='"')
        writer.writerows(datas)
        
See ``QUOTE_NONNUMERIC`` option for automatic conversion of non-quoted fields to float.

Alternatively lines can be read as dicts with fieldnames using `DictReader <http://docs.python.org/3/library/csv.html#csv.DictReader>`_ and `DictWriter <http://docs.python.org/3/library/csv.html#csv.DictWriter>`_
::
    reader=csv.DictReader(csvfile, fieldnames=("a", "b", "c")) # rows are read into dicts with these key
    # omit fieldnames if first line of csvfile determines key
    # additional fields can be dumped into a key determined by a restval parameter
    writer=csv.DictWriter(csvfile, fieldnames)

Analysis
--------
Curve fitting
.............
One can use the function `scipy.optimize.curve_fit <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_
::
    popt, pcov = curve_fit.(func, xdata, ydata)  # ydata = f(xdata, *params) + eps