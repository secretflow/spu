# Examples

This directory contains examples demonstrating how to use SPU to write privacy-preserving machine learning programs.

* [ss_lr](ss_lr/): Private training of linear/logistic regression models with JAX.
* [ss_xgb](ss_xgb/): Private training of an XGB model with JAX.
* [jax_lr](jax_lr/): Private training of a logistic regression model with JAX (different from `ss_lr`,
                     we run a complete JAX training function in this demo).
* [jax_svm](jax_svm/): Private training of an SVM model with JAX.
* [jax_kmeans](jax_kmeans/): Private training of K-Means clustering with JAX.
* [flax_mlp](flax_mlp/): Private training of an MLP model with [Flax](https://github.com/google/flax).
* [stax_nn](stax_nn/): Private training of four widely-evaluated neural networks for MNIST classification with
                       [Stax](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html).
* [stax_mnist_classifier](stax_mnist_classifier/): Private training of a simple neural network for MNIST classification with
                                                   [Stax](https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html).
* [flax_resnet](flax_resnet/): Private training of a [ResNet](https://arxiv.org/abs/1512.03385) model with [Flax](https://github.com/google/flax) library.
* [flax_gpt2](flax_gpt2/): Private inference of a pre-trained
                           [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
                           model with [Flax](https://github.com/google/flax) library.
* [flax_vae](flax_vae/): Private training of a [VAE](http://arxiv.org/abs/1312.6114) model for MNIST digit reconstruction with
                         [Flax](https://github.com/google/flax) library.
* [haiku_lstm](haiku_lstm/): Private training of an LSTM model with [Haiku](https://github.com/deepmind/dm-haiku).
* [jraph_gnn](jraph_gnn/): Private training of a [graph convolutional network](https://arxiv.org/abs/1609.02907) model with
                           [Jraph](https://github.com/deepmind/jraph).
* [tf_experiment](tf_experiment/): Private training of a logistic regression model with TensorFlow (**experimental**).
* [torch_experiment](torch_experiment/): Private inference of a linear regression model with PyTorch (**experimental**).
