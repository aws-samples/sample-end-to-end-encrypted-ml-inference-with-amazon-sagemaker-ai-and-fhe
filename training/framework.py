# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from sagemaker.estimator import Framework

from common import constants


class Concrete(Framework):
    def __init__(
        self,
        entry_point,
        source_dir=None,
        hyperparameters=None,
        py_version="py3",
        framework_version="1.5.0",
        distributions=None,
        **kwargs,
    ):
        image_uri = self._make_image_uri(constants.ECR_REPO_TRAINING)
        super(Concrete, self).__init__(
            entry_point, source_dir, hyperparameters, image_uri=image_uri, **kwargs
        )
        self.framework_version = framework_version
        self.py_version = py_version

    def _make_image_uri(self, repo):
        image_uri = "%s/%s:latest" % (constants.ECR_REGISTRY, repo)
        return image_uri

    def training_image_uri(self, region=None):
        image_uri = self._make_image_uri(constants.ECR_REPO_TRAINING)
        return image_uri

    def create_model(
        self,
        model_server_workers=None,
        role=None,
        vpc_config_override=None,
        entry_point=None,
        source_dir=None,
        dependencies=None,
        image_name=None,
        **kwargs,
    ):
        return None
