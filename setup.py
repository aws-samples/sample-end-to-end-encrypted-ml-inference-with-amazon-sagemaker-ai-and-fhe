# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import base64
import logging
import os
import pathlib
import shutil
import sys

import docker
import numpy as np
from joblib import Memory
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from common import constants
from common.sessions import create_session

LOGGER = logging.getLogger(__name__)

script_dir = pathlib.Path(__file__).resolve().parent


def setup_data(
    caching_dir,
    training_features_path,
    training_labels_path,
    testing_features_path,
    testing_labels_path,
    model_output_dir,
):

    LOGGER.info("Setting up data")
    data_paths = [
        training_features_path,
        training_labels_path,
        testing_features_path,
        testing_labels_path,
    ]

    # Remove old files if they exist
    for path in data_paths:
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists(model_output_dir):
        shutil.rmtree(model_output_dir)

    # Create directories if needed
    pathlib.Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    for p in data_paths:
        dir_name = os.path.dirname(p)
        if not os.path.exists(dir_name):
            pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)

    # scikit-learn's fetch_openml method doesn't handle local cache:
    # https://github.com/scikit-learn/scikit-learn/issues/18783#issuecomment-723471498
    # This is a workaround that prevents downloading the data every time the notebook is ran
    memory = Memory(caching_dir)
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist_dataset = fetch_openml_cached("mnist_784")

    data = mnist_dataset.data
    target = mnist_dataset.target

    # Normalization required by concrete.
    # Define max, mean and std values for the MNIST data set
    max_value = 255
    mean = 0.1307
    std = 0.3081

    # Normalize the training and test features
    data = data / max_value
    data = ((data - mean) / std).round(decimals=4)

    data = data.to_numpy()
    target = target.to_numpy(dtype=np.int64)

    test_size = 10000
    features_train, features_test = train_test_split(
        data, test_size=test_size, random_state=0
    )
    labels_train, labels_test = train_test_split(
        target, test_size=test_size, random_state=0
    )

    for array, path in [
        (features_train, training_features_path),
        (labels_train, training_labels_path),
        (features_test, testing_features_path),
        (labels_test, testing_labels_path),
    ]:
        with open(path, mode="wb") as f:
            np.save(f, array)


def setup_buckets(
    data_bucket_name,
    model_bucket_name,
    training_features_path,
    training_labels_path,
    s3_prefix,
    delete_old_models=False,
):

    LOGGER.info("Setting up buckets")

    session = create_session()
    s3 = session.resource("s3")

    #
    # Training data
    #

    bucket = s3.Bucket(data_bucket_name)

    # Empty bucket of MNIST data
    bucket.objects.filter(Prefix=s3_prefix).delete()

    # Put files in bucket
    for path in [training_features_path, training_labels_path]:
        basename = os.path.basename(path)
        s3_key = s3_prefix + "/" + basename
        with open(path, "rb") as f:
            bucket.put_object(Body=f, Key=s3_key, ServerSideEncryption="AES256")

    #
    # Model
    #

    bucket = s3.Bucket(model_bucket_name)
    if delete_old_models:
        # Empty bucket of MNIST data
        bucket.objects.filter(Prefix=s3_prefix).delete()


def setup_containers(
    ecr_registry,
    ecr_training_repo,
    ecr_inference_repo,
    training_container_only,
    inference_container_only,
):

    LOGGER.info("Setting up containers")

    session = create_session()
    ecr = session.client("ecr")
    token = ecr.get_authorization_token()
    encoded_password = token["authorizationData"][0]["authorizationToken"]
    username_password = base64.standard_b64decode(encoded_password).decode("ascii")
    assert len(username_password.split(":")) == 2
    username = username_password.split(":")[0]
    password = username_password.split(":")[1]
    docker_client = docker.from_env()

    # Build and push containers
    container_configs = []
    if not inference_container_only:
        container_configs += [("training", "Dockerfile.training", ecr_training_repo)]
    if not training_container_only:
        container_configs += [
            ("inference/endpoint", "Dockerfile.inference", ecr_inference_repo)
        ]

    for dockerfile_path, dockerfile_name, ecr_repo in container_configs:
        LOGGER.debug("Building %s", dockerfile_name)
        this_build_context = os.path.join(script_dir, dockerfile_path)
        this_dockerfile_path = os.path.join(this_build_context, dockerfile_name)
        ecr_tag = "%s/%s:latest" % (ecr_registry, ecr_repo)
        LOGGER.debug("ECR tag: {}".format(ecr_tag))
        image, logs = docker_client.images.build(
            path=this_build_context,
            dockerfile=this_dockerfile_path,
            tag=ecr_tag,
            nocache=True,
        )
        for logline in logs:
            LOGGER.debug(logline)
        # Push container to ECR
        LOGGER.debug("Starting tag")
        resp = docker_client.login(
            username=username, password=password, registry=ecr_registry
        )
        LOGGER.debug("Pushing image")
        resp = docker_client.images.push(
            ecr_tag, auth_config={"username": username, "password": password}
        )
        LOGGER.debug(resp)


def main():
    parser = argparse.ArgumentParser(
        prog="setup.py",
        description="Sets up your local and AWS environments for running the experiments in this directory. Requires the user to uncommented and filled in the missing entries in constants.py",
    )
    parser.add_argument(
        "-d", "--data-only", help="Set up local data only", action="store_true"
    )
    parser.add_argument(
        "-b",
        "--buckets-only",
        help="Set up buckets only (requires data be present locally)",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--training-container-only",
        help="Set up training container image only",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--inference-container-only",
        help="Set up inference container image only",
        action="store_true",
    )

    args = parser.parse_args()
    onlys = [
        args.buckets_only,
        args.data_only,
        args.training_container_only,
        args.inference_container_only,
    ]
    num_onlys = sum([1 if x else 0 for x in onlys])
    if num_onlys > 1:
        sys.exit(
            "Only one of --buckets-only, --data-only, --training-containers-only, and --inference-containers-only can be used."
        )

    LOGGER.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(filename)s:%(name)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    LOGGER.info("starting")

    if not (args.buckets_only or args.data_only):
        setup_containers(
            constants.ECR_REGISTRY,
            constants.ECR_REPO_TRAINING,
            constants.ECR_REPO_INFERENCE,
            args.training_container_only,
            args.inference_container_only,
        )

    if not (
        args.buckets_only
        or args.training_container_only
        or args.inference_container_only
    ):
        setup_data(
            constants.MNIST_DIR,
            constants.MNIST_TRAINING_FEATURES_PATH,
            constants.MNIST_TRAINING_LABELS_PATH,
            constants.MNIST_TESTING_FEATURES_PATH,
            constants.MNIST_TESTING_LABELS_PATH,
            constants.MNIST_MODEL_OUTPUT_DIR,
        )

    if not (
        args.data_only or args.training_container_only or args.inference_container_only
    ):
        setup_buckets(
            constants.DATA_BUCKET_NAME,
            constants.MODEL_BUCKET_NAME,
            constants.MNIST_TRAINING_FEATURES_PATH,
            constants.MNIST_TRAINING_LABELS_PATH,
            constants.MNIST_PREFIX,
        )

    LOGGER.info("Done")


if __name__ == "__main__":
    main()
