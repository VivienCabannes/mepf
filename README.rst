ENTSEARCH: Mode search through entropic coding
==============================================

Installation
------------
From source
~~~~~~~~~~~
Once download, our packages can be install through the following command.

.. code:: shell

   $ cd <path to code folder>
   $ pip install -e .

Notes
-----
The code is intended to test research ideas, not to be used in production.
In particular, it was coded assuming that we were omniscient of the labels, which simplified some design choices.

To minimize boilerplate code, we provide a single class that can be called to run any algorithm.
However, to improve code readibility, each algorithm is equally implemented in separate files.
