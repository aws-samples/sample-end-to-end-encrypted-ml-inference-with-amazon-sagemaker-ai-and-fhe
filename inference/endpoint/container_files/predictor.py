# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import hashlib
import json
import logging
import os
import sys

import flask
from concrete.ml.deployment import FHEModelServer
from flask import Flask
from sagemaker.s3 import S3Downloader

LOGGER = logging.getLogger(__name__)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

# quiet down some noisy loggers
for noisy_module in ["botocore", "s3transfer", "boto3", "urllib3"]:
    noisy_module_handler = logging.getLogger(noisy_module)
    noisy_module_handler.setLevel(logging.INFO)

# Load in model
# Model artifacts should be stored in /opt/ml/model/
try:
    model = FHEModelServer("/opt/ml/model/")
    LOGGER.debug("Successfully initialized FHEModelServer")
except Exception:
    LOGGER.exception("Failed to initialize FHEModelServer")
    raise

# The flask app for serving predictions
app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    # Called by Amazon SageMaker AI to confirm container health
    return flask.Response(response="\n", status=200, mimetype="application/json")


def _hash_string_to_hex(input_string):
    encoded_string = input_string.encode("utf-8")
    sha256_hash = hashlib.sha256()
    sha256_hash.update(encoded_string)
    hex_digest = sha256_hash.hexdigest()
    return hex_digest


def _get_evaluation_key(evaluation_keys_uri):
    hash_key = _hash_string_to_hex(evaluation_keys_uri)
    if not os.path.exists("/tmp/evaluation_keys"):
        os.mkdir("/tmp/evaluation_keys")
    if hash_key not in os.listdir("/tmp/evaluation_keys"):
        LOGGER.debug(f"Do not have evaluation keys for {evaluation_keys_uri}")
        downloader = S3Downloader()
        evaluation_keys = downloader.read_bytes(evaluation_keys_uri)
        with open("/tmp/evaluation_keys/" + hash_key, "wb") as f:
            f.write(evaluation_keys)
        LOGGER.debug(f"Downloaded evaluation keys from {evaluation_keys_uri}")
    with open(os.path.join("/tmp/evaluation_keys", hash_key), "rb") as f:
        return f.read()


@app.route("/invocations", methods=["POST"])
def transformation():
    LOGGER.debug("calling transformation()")
    try:
        input_json = flask.request.get_json()
        LOGGER.debug("input_json: {}".format(input_json))

        if not input_json or not isinstance(input_json, dict):
            return flask.Response(
                response=json.dumps({"error": "Invalid JSON"}),
                status=400,
                mimetype="application/json",
            )

        required_keys = [
            "evaluation_keys_uri",
            "encrypted_query_uri",
        ]
        for key in required_keys:
            if key not in input_json:
                return flask.Response(
                    response=f"Missing required field: {key}", status=400
                )
            if not isinstance(input_json[key], str) or not input_json[key].startswith(
                "s3://"
            ):
                return flask.Response(response=f"Invalid S3 URI for {key}", status=400)

        evaluation_keys_uri = input_json["evaluation_keys_uri"]
        LOGGER.debug("evaluation_keys_uri: {}".format(evaluation_keys_uri))

        encrypted_query_uri = input_json["encrypted_query_uri"]
        LOGGER.debug("encrypted_query_uri: {}".format(encrypted_query_uri))

        downloader = S3Downloader()
        LOGGER.debug("Created downloader")

        try:
            evaluation_keys = _get_evaluation_key(evaluation_keys_uri)
            encrypted_query = downloader.read_bytes(encrypted_query_uri)
            LOGGER.debug("Downloaded encrypted query")
        except Exception as e:
            LOGGER.error(f"Failed to download from S3: {e}")
            return flask.Response(
                response="Failed to retrieve data from S3", status=500
            )

        # Output is an encrypted blob that only the client can decrypt; no server-side filtering is possible
        prediction = model.run(encrypted_query, evaluation_keys)
        LOGGER.debug("Computed prediction")

        return flask.Response(
            response=prediction, status=200, mimetype="application/octet-stream"
        )

    except Exception:
        return flask.Response(
            response=json.dumps({"error": "Internal server error"}),
            status=500,
            mimetype="application/json",
        )
