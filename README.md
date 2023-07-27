# Mean-Field Multi-Agent Contextual Bandit for Energy-Efficient Resource Allocation in vRANs

Implementation for IEEE INFOCOM'24 of the paper: [Mean-Field Multi-Agent Contextual Bandit for Energy-Efficient Resource Allocation in vRANs](#) 

## Dependencies
* Python >= 3.8
* [Pytorch](https://pytorch.org/) >= 1.13.0 
* [numpy](https://numpy.org/) >= 1.21
* gym >= 0.26

## File Structure
- `eval_learning_alg_mf.py` is the main file that loads the learning algorithm, configures the hyperparameters, and connects with the O-RAN platform to run the experiments.
- `learning_algorithms_mf.py` implements the mean field multi-agent contextual bandit algorithm, including the computation of the mean field approximation, and the implementation of the gradient update.
- `models.py` includes the classes that implement the models used by the learning algorithm, i.e., MLP, 3D convolutional networks, and a class to build the actor and critic.
- `models_utils.py` includes some auxilary classes used by the models.
- `process_data_classes.py` includes a set of classes used to process the data gathered from the experimental platform in order to feed the learning algorithm.
- `utils_learning.py` includes a set of classes that implement the exploration noise and the reply buffers used by the learning algorithm.
 
This repository will be publicly available on GitHub upon the acceptance of the paper.





