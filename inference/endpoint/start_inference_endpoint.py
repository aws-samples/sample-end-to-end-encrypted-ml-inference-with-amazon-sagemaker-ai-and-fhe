# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import logging

from sagemaker.async_inference.async_inference_config import \
    AsyncInferenceConfig
from sagemaker.model import Model
from sagemaker.predictor import Predictor

from common.constants import (ECR_REGISTRY, ECR_REPO_INFERENCE,
                              ENDPOINT_EXECUTION_ROLE, MNIST_PREFIX,
                              MODEL_BUCKET_NAME, QUERY_BUCKET_NAME)
from common.sessions import create_sagemaker_session, create_session
from inference.inference_common import get_model_location

LOGGER = logging.getLogger(__name__)


def start_endpoint(session, model_location, inference_image_uri):
    # deploy model
    LOGGER.info("deploying model")
    sagemaker_session = create_sagemaker_session(session)
    model = Model(
        image_uri=inference_image_uri,
        model_data=model_location,
        role=ENDPOINT_EXECUTION_ROLE,
        sagemaker_session=sagemaker_session,
        predictor_cls=Predictor,
    )
    output_path = "s3://" + QUERY_BUCKET_NAME + "/async/output"
    failures_path = "s3://" + QUERY_BUCKET_NAME + "/async/failures"
    async_config = AsyncInferenceConfig(
        max_concurrent_invocations_per_instance=1,
        output_path=output_path,
        failure_path=failures_path,
    )
    endpoint = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        wait=True,
        endpoint_logging=True,
        async_inference_config=async_config,
        encrypt_inter_container_traffic=True,
        endpoint_name="concrete-fhe-" + MNIST_PREFIX.lower(),
    )
    return endpoint


if __name__ == "__main__":
    import sys

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

    model_location = get_model_location(session, MODEL_BUCKET_NAME, MNIST_PREFIX)
    endpoint = start_endpoint(session, model_location, image_uri)
    print(f"Endpoint name: {endpoint.endpoint_name}")
