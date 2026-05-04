# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from __future__ import print_function

import argparse
import logging
import os

import numpy
from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn import NeuralNetClassifier
from torch import nn

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


def do_training(model_dir, train):

    # Input the feature and label arrays
    feature_file = os.path.join(train, "mnist_training_features.npy")
    x_train = numpy.load(feature_file)
    label_file = os.path.join(train, "mnist_training_labels.npy")
    y_train = numpy.load(label_file)
    assert x_train.shape[1] == 784, f"Unexpected feature dimensions: {x_train.shape}"
    assert len(y_train) == len(x_train), "Feature/label count mismatch"

    # Model taken from Concrete's example code: https://github.com/zama-ai/concrete-ml/blob/release/1.9.x/docs/advanced_examples/FullyConnectedNeuralNetworkOnMNIST.ipynb
    params = {
        "module__n_layers": 2,
        "module__n_w_bits": 4,
        "module__n_a_bits": 4,
        "module__n_hidden_neurons_multiplier": 0.5,
        "module__activation_function": nn.ReLU,
        "max_epochs": 7,
    }

    LOGGER.info("Training model")
    model = NeuralNetClassifier(**params)
    model.fit(X=x_train, y=y_train)
    LOGGER.info("Compiling model")
    model.compile(x_train)

    FHEModelDev(model_dir, model).save()


def model_fn(model_dir):
    """Deserialized and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    # Not used: model deserialization is handled by FHEModelServer in the inference container
    raise NotImplementedError


print("training_script.py loaded")

if __name__ == "__main__":
    print("Main function called on training_script.py")
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    #    parser.add_argument('--activation', type=str, default="logistic")

    # Amazon SageMaker AI specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    args = parser.parse_args()

    import sys

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    do_training(args.model_dir, args.train)
