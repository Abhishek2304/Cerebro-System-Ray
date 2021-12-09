Cerebro On Ray
==============
 
``Cerebro`` is a data system for optimized deep learning model selection. It uses a novel parallel execution strategy
called **Model Hopper Parallelism (MOP)** to execute end-to-end deep learning model selection workloads in a more 
resource-efficient manner. Detailed technical information about ``Cerebro`` can be found in the 
[Technical Report](https://adalabucsd.github.io/papers/TR_2020_Cerebro.pdf).


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

For the implementation, we make changes to the following folders: 

**cerebro/backend**: We add another folder cerebro/backend/ray which contains the entire implementation of the Ray backend.
**cerebro/keras**: We add another folder cerebro/keras/ray which contains the entire implementation of the Keras Estimator used by the Ray backend.
**cerebro/examples**: We add 2 examples: **cerebro/examples/mnist** that runs the model on an augmented MNIST dataset 
                                     **cerebro/examples/criteo** that runs the model on one partition of the Criteo dataset


Examples
-------------
We provide 2 detailed examples of Cerebro in the examples folder: mnist and criteo (the deep postures and other scripts correspond to Spark)
The details on running these examples is found in the READMEs of each such folder
