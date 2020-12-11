Installation
============

Github Installation
-------------------
Go to the directory from which you want to run this package, and then open a command line prompt. 

* Visit `GradDog <https://github.com/git-fetch-git-roll-over/GradDog.git>`__ and follow the cloning instructions to clone a copy of the repository. ``git clone https://github.com/git-fetch-git-roll-over/GradDog.git``  will create an ``GradDog`` directory.
* ``cd GradDog`` to go inside the directory.
* ``virtualenv graddog`` to create a virtual environment. (It is optional but recommended to use a virtual environment.)
* ``source graddog/bin/activate`` to activate the environment.
* ``pip install -r requirements.txt`` to install the necessary dependencies.
* create your own driver script and follow some demostrations in the basic demos section.



Pip Installation
----------------
``GradDog`` is also pip-installable. We used `wheels <https://www.python.org/dev/peps/pep-0427/>`_ to package our project. The wheel name for our package is ``graddog-1.3-py3-none-any.whl``.
To use pip, simply type ``pip install graddog``.
