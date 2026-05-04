# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
import datetime
import logging
import os
import sys
import tarfile
import tempfile

import numpy
from concrete.ml.deployment import FHEModelClient
from sagemaker.async_inference.waiter_config import WaiterConfig
from sagemaker.base_deserializers import BytesDeserializer
from sagemaker.base_serializers import JSONSerializer
from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.s3 import S3Downloader, S3Uploader
from sklearn.metrics import accuracy_score

from common.constants import (ECR_REGISTRY, ECR_REPO_INFERENCE, MNIST_PREFIX,
                              MNIST_TESTING_FEATURES_PATH,
                              MNIST_TESTING_LABELS_PATH, MODEL_BUCKET_NAME,
                              QUERY_BUCKET_NAME)
from common.sessions import create_sagemaker_session, create_session
from inference.inference_common import get_model_location

LOGGER = logging.getLogger(__name__)


def get_prediction(
    predictor, concrete_client, session, this_sample, eval_keys_s3_uri, s3_directory, i
):

    this_query_s3_dir = s3_directory + "/{:0>8d}/".format(i)
    encrypted_query_uri = this_query_s3_dir + "encrypted_query"
    query_input_path_uri = this_query_s3_dir + "query"

    query = {
        "evaluation_keys_uri": eval_keys_s3_uri,
        "encrypted_query_uri": encrypted_query_uri,
    }

    LOGGER.debug("query: %s", query)

    encrypted_query = concrete_client.quantize_encrypt_serialize(this_sample)
    LOGGER.debug("sample encrypted")
    uploader = S3Uploader()
    uploader.upload_bytes(
        encrypted_query, encrypted_query_uri, sagemaker_session=session,
    )
    LOGGER.debug("encrypted query uploaded")
    try:
        async_response = predictor.predict_async(
            data=query,
            input_path=query_input_path_uri,
            initial_args={"ContentType": "application/json"},
        )
        LOGGER.debug("Query submitted. Waiting on result.")

        # Wait for result from endpoint
        encrypted_result = async_response.get_result(
            waiter_config=WaiterConfig(max_attempts=120, delay=30) 
        )
        LOGGER.debug("Encrypted result received")

        prediction_array = concrete_client.deserialize_decrypt(encrypted_result)
        LOGGER.debug("Prediction decrypted: %s", prediction_array)
        prediction = prediction_array.argmax()
        LOGGER.debug("prediction: %s", prediction)
        return prediction
    except TimeoutError:
        LOGGER.error("Timeout waiting for prediction result")
        raise
    except Exception:
        LOGGER.error("Error getting prediction", exc_info=True)
        raise


def run_inference(
    session,
    predictor,
    model_location,
    testing_features_path,
    testing_labels_path,
    query_bucket,
):
    # Load data:
    x_test = numpy.load(testing_features_path)
    y_test = numpy.load(testing_labels_path)
    assert x_test.shape[1] == 784, f"Unexpected feature dimensions: {x_test.shape}"
    assert len(y_test) == len(x_test), "Feature/label count mismatch"

    # get client_config from S3
    sagemaker_session = create_sagemaker_session(session)
    with tempfile.TemporaryDirectory() as config_dir_name:
        LOGGER.debug(config_dir_name)
        try:
            S3Downloader().download(
                model_location,
                local_path=config_dir_name,
                sagemaker_session=sagemaker_session,
            )
            LOGGER.debug(config_dir_name)
            tf = tarfile.open(
                os.path.join(config_dir_name, "model.tar.gz"), mode="r:gz"
            )
            LOGGER.debug(config_dir_name)
            tf.extract("client.zip", config_dir_name)
            LOGGER.debug(config_dir_name)
        except FileNotFoundError:
            LOGGER.error("Model file not found in %s", config_dir_name)
            raise
        except tarfile.TarError:
            LOGGER.error("Failed to extract model archive", exc_info=True)
            raise
        except Exception:
            LOGGER.error("Error loading model config", exc_info=True)
            raise

        with tempfile.TemporaryDirectory() as key_dir_name:
            concrete_client = FHEModelClient(config_dir_name, key_dir=key_dir_name)

            # Generate keys, store in S3
            concrete_client.generate_private_and_evaluation_keys()
            date_format = "%Y-%m-%d-%H-%M-%S-%f"
            s3_prefix = "MNIST/" + datetime.datetime.now().strftime(date_format)
            s3_directory = "s3://" + query_bucket + "/" + s3_prefix
            eval_keys_s3_uri = s3_directory + "/evaluation_keys"
            eval_keys = concrete_client.get_serialized_evaluation_keys()
            uploader = S3Uploader()
            uploader.upload_bytes(
                eval_keys, eval_keys_s3_uri, sagemaker_session=sagemaker_session,
            )

            # Ask for some predictions
            num_samples_to_send = 10
            predictions = []
            assert num_samples_to_send <= x_test.shape[1]
            for i in range(num_samples_to_send):
                start_time = datetime.datetime.now()
                this_sample = x_test[i : i + 1, :]
                LOGGER.debug(f"Starting query {i}")
                prediction = get_prediction(
                    predictor,
                    concrete_client,
                    sagemaker_session,
                    this_sample,
                    eval_keys_s3_uri,
                    s3_directory,
                    i,
                )
                LOGGER.info("Prediction received: %s", prediction)
                this_label = y_test[i : i + 1]
                LOGGER.info("Correct label: %s", this_label)
                predictions.append(prediction)
                end_time = datetime.datetime.now()
                elapsed_time = end_time - start_time
                LOGGER.info("Elapsed_time: %s", elapsed_time)

            labels = y_test[0:num_samples_to_send]
            # Calculate accuracy
            accuracy_metrics = accuracy_score(labels, predictions)
            LOGGER.info("accuracy score: %s", accuracy_metrics)


def get_predictor(endpoint_name, session):
    sagemaker_session = create_sagemaker_session(session)
    sync_predictor = Predictor(
        endpoint_name,
        serializer=JSONSerializer(),
        deserializer=BytesDeserializer(),
        sagemaker_session=sagemaker_session,
    )
    return AsyncPredictor(sync_predictor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint_name")
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s: %(filename)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    image_uri = "%s/%s:latest" % (ECR_REGISTRY, ECR_REPO_INFERENCE)
    session = create_session()
    endpoint_name = args.endpoint_name
    predictor = get_predictor(endpoint_name, session)

    model_location = get_model_location(session, MODEL_BUCKET_NAME, MNIST_PREFIX)

    run_inference(
        session,
        predictor,
        model_location,
        MNIST_TESTING_FEATURES_PATH,
        MNIST_TESTING_LABELS_PATH,
        QUERY_BUCKET_NAME,
    )
