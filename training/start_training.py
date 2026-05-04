# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import logging

from common.constants import (DATA_BUCKET_NAME, MNIST_PREFIX,
                              MODEL_BUCKET_NAME, MODEL_TRAINING_ROLE,
                              SCRIPT_BUCKET_NAME)
from common.sessions import create_sagemaker_session, create_session
from training.framework import Concrete

LOGGER = logging.getLogger(__name__)


def run_training(
    s3_prefix, data_bucket_name, model_bucket_name, base_job_name, script_bucket_name
):

    session = create_session()
    sagemaker_session = create_sagemaker_session(session)

    script_path = "training_script.py"
    training_location = "s3://" + data_bucket_name + "/" + s3_prefix
    model_location = "s3://" + model_bucket_name + "/" + s3_prefix
    training_script_location = "s3://" + script_bucket_name + "/" + s3_prefix

    concrete = Concrete(
        entry_point=script_path,
        instance_count=1,
        instance_type="ml.c5.2xlarge",
        role=MODEL_TRAINING_ROLE,
        sagemaker_session=sagemaker_session,
        hyperparameters={},
        output_path=model_location,
        base_job_name=base_job_name,
        code_location=training_script_location,
        encrypt_inter_container_traffic=True,
    )

    concrete.fit(inputs=training_location)


if __name__ == "__main__":
    run_training(
        MNIST_PREFIX,
        DATA_BUCKET_NAME,
        MODEL_BUCKET_NAME,
        MNIST_PREFIX,
        SCRIPT_BUCKET_NAME,
    )
