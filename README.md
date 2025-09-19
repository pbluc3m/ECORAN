# Mean-Field Multi-Agent Contextual Bandit for Energy-Efficient Resource Allocation in vRANs

Source code of the paper [Mean-Field Multi-Agent Contextual Bandit for Energy-Efficient Resource Allocation in vRANs](#) published in the proceedings of IEEE INFOCOM'24.

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

## Acknowledgements

This work is partly supported by the Spanish Ministry of Economic Affairs and Digital Transformation and the European Union-NextGenerationEU through the [UNICO 5G I+D SORUS project](https://unica6g.it.uc3m.es/en/6g-sorus/).

![PRTR](prtr.png)
