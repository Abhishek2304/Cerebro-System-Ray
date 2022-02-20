Ray Backend for Cerebro
=======================
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in the 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf). \
We present a backend of this system deployed on Python's scalable systems library: Ray.


Install
-------

You can take the script from this repo called cloudlab_init.sh and run it in a bash shell

    bash cloudlab_init.sh

The script will install all dependencies (Linux and Python) and make a project directory in your current folder. It will clone and make the cerebro-system repo in this folder and install a Python virtual environment env_cerebro. You need to activate the environment to run the scripts. Go to the project folder and run:

    source env_cerebro/bin/activate or activate.csh

You MUST be running on **Python >= 3.6** with **Tensorflow == 2.3** (The script should install the appropriate Python and tensorflow for you)


Documentation
-------------

Detailed documentation about the system can be found [here](https://adalabucsd.github.io/cerebro-system/).

Implementation
--------------

For our implementation of the Ray backend, the source codes can be found in the following folders: 

**cerebro/backend**: We add another folder cerebro/backend/ray which contains the entire implementation of the Ray backend. \
**cerebro/keras**: We add another folder cerebro/keras/ray which contains the entire implementation of the Keras Estimator used by the Ray backend. \
**cerebro/examples**: We add 1 example: **cerebro/examples/mnist** that runs the model on an augmented MNIST dataset.

Examples
-------------
We provide a detailed examples of running our implementation of Ray on a large augmented MNIST dataset and comparing with Sequential Keras and Ray Tune in the examples folder: mnist (the deep postures and other scripts correspond to Spark).
The details on running this example is found in the README of the folder.
