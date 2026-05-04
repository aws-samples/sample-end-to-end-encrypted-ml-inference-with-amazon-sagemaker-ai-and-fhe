# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import boto3
from sagemaker import Session as SageMakerSession

from common.constants import REGION_NAME


def create_session():
    session = boto3.session.Session(region_name=REGION_NAME)
    return session


def create_sagemaker_session(boto_session):
    return SageMakerSession(boto_session=boto_session)
