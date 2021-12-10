
# Background
We provide an example of how to use `Cerebro-Ray` to scale the model selection on an augmented version of the MNIST data.

# Steps
## 1. Preparing the Training Data
1. Get the training and validation data from here: https://drive.google.com/drive/folders/1qMU3MhbLULxXrm8ZM0gh4A3UruVV9SJO?usp=sharing

## 2. (Optional) Modify the Workload
1. The workload is defined in the `model_selection.py` script. There are three main componenets: 1) `param_grid_criteo` definition in the main function, 2) `estimator_gen_fn` function and 3) `main` function
2. `param_grid_criteo` is a dictionary defining the tuning parameters. Currently, it defines 2 different values for learning rate and regularization value each. You can add more parameters values if you need.
3. `estimator_gen_fn` takes in a parameter value instance and initilizes a `cerebro.RayEstimator` object, which is ready to be trained.
4. The `main` function encapsulates the model selection procedure. The script uses `GridSearch`. However, you can also use `RandomSearch` or `TPESearch`.

## 3. Running the workload
1. The data has already been prepared in the format that Cerebro Ray takes. Look at this dataset (Column names, ways in which columns are represented) for reference to prepare other data for our implementation
2. Where the comments are given in the model_selection.py script, use it to change the root folders where you store data, the train and val data filenames, the hyperparameters to train on, and the model definition.
3. After running the `cloudlab_init.sh` script, if you want to run Ray on only one machine, you can run the model_selection.py script directly, by calling `python3 model_selection.py`.
4. To run ray on multiple machines, run `ray start --head` on the master node. You will get a command of the form:
   `ray start --address=<address_name> --redis-password=<redis_password>`
   Copy this command and run it on all machines you want the Ray cluster on.
   Then run the script on the master node by running the command `python3 model_selection.py`.
   
5. The other scripts `test_single.py` and `tune_selection.py` are for comparing our approach to a sequential approach and Ray Tune.
