Software organization
=====================

Directory Structure
-------------------

With our main directory ``GradDog``, our package is organized as follows:

* dependencies in ``requirements.txt``
* continuous integration information in ``.travis.yml`` and coverage information in ``codecov.yml``
* documentation in ``\docs``
* test suite in ``\tests``
* code in ``\graddog``
* images in ``\images``

Basic Modules
-------------

**For users:**
    
Our code consists of 3 user-facing modules: 

* ``graddog``, defining the ``trace`` function used to return derivatives
* ``graddog.functions``, definining elementary functions that are compatible with the trace function.
* ``graddog.tools``, defining useful tools to plot derivatives and estimate extrema.
    
**For developers:**

Our code consists of 5 total modules:

* ``__init__.py``, defining the trace function that allows the user to take derivatives
* ``trace.py``, defining the Trace class that stores the value and derivative for each element
* ``compgraph.py``, defining the CompGraph class that creates a computational graph keeping track of the elements in relation to one another
* ``functions.py``, defining mathematical functions that operate on the Trace class 
* ``math.py``, defining the mathematical rules that define each functions derivatives
* ``tools.py``, defining user-facing tools to do perform common calculations

How to Test
------------

Both ``TravisCI`` and ``CodeCov`` are used and our test suite will live inside our repo in a ``\tests`` directory. Our project currently has over 90% coverage.
