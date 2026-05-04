# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import logging

LOGGER = logging.getLogger(__name__)


def get_model_location(session, model_bucket_name, s3_prefix):
    s3 = session.resource("s3")
    bucket = s3.Bucket(model_bucket_name)
    bucket_objects = bucket.objects.filter(Prefix=s3_prefix).all()
    most_recent_subdir = None
    model_s3_path = None
    for o in bucket_objects:
        # Find the o corresponding to the most recent model:
        parts = o.key.split("/")
        if len(parts) == 4:
            if all(
                [
                    parts[0] == s3_prefix,
                    parts[2] == "output",
                    parts[3] == "model.tar.gz",
                    (most_recent_subdir is None or most_recent_subdir > parts[2]),
                ]
            ):
                most_recent_subdir = parts[2]
                model_s3_path = o.key

    if model_s3_path is None:
        raise ValueError(f"No model found in s3://{model_bucket_name}/{s3_prefix}")
    LOGGER.debug("model_s3_path: %s", model_s3_path)
    model_location = "s3://" + model_bucket_name + "/" + model_s3_path
    return model_location
