Software organization
=====================

Directory Structure
-------------------

With our main directory ``GradDog``, our package is organized as follows:

* dependencies in ``requirements.txt``
* workflow for continuous integration via ``GitHub Actions`` and code coverage report via ``CodeCov`` in ``.github\workflow\python-package.yml``
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
* ``graddog.tools``, defining user-facing tools to do common calculations of scalar functions
    
**For developers:**

Our code consists of 5 total modules:

* ``__init__.py``, defining the trace function that allows the user to take derivatives of python functions
* ``trace.py``, defining the Trace class that stores the information for an individual trace element
* ``compgraph.py``, defining the CompGraph class that creates a computational graph keeping track of the elements in relation to one another
* ``functions.py``, defining elementary functions that users can use
* ``math.py``, defining the mathematical rules (for computing values, first-derivatives, and second-derivatives) of all functions implemented in our library
* ``tools.py``, defining user-facing tools to do common calculations of scalar functions

How to Test
------------

We used ``GitHub Actions`` for continuous integration and ``CodeCov`` for code coverage reports. Our test suite will live inside our repo in a ``\tests`` directory. Our project currently has 98% coverage. In order to run the tests locally after installing the repository, run the following command in your terminal while being in the root directory::

    pytest 

If you want to see code coverage, run the following commands in your terminal::
    
    pip install pytest-cov
    pytest --cov-report term --cov=graddog tests/
    
    
    
