## End to End Encrypted ML Inference with Amazon Sagemaker AI and FHE

This project contains sample code for a forthcoming blog post. 

## Security

This solution uses multiple Amazon Web Services (AWS) services and follows the AWS shared responsibility model
(https://aws.amazon.com/compliance/shared-responsibility-model/).

AWS responsibilities (security OF the cloud):
- Physical security of data centers and underlying infrastructure
- Managed service infrastructure for Amazon SageMaker AI, Amazon Simple Storage Service (Amazon S3),
  Amazon Elastic Container Registry (Amazon ECR), and AWS Identity and Access Management (IAM).
- Service availability, patching, and network infrastructure

Your responsibilities (security IN the cloud):
- IAM role and policy configuration with least-privilege access
- Amazon S3 bucket encryption, access controls, and Block Public Access settings
- Container image security and base image updates
- Network configuration (Amazon Virtual Private Cloud (Amazon VPC), security groups) if applicable
- Data encryption in transit and at rest
- Monitoring and audit logging via Amazon CloudWatch and AWS CloudTrail
- Regular review of IAM policies and removal of unused permissions

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Notes

SageMaker AI training jobs and endpoints incur charges based on instance type and duration.
Endpoints continue to incur charges until deleted. Remember to clean up resources when finished to
avoid unnecessary costs. See the Clean up resources section for details.

Also, you are responsible for reviewing and complying with the licenses applicable to the dependencies
in the requirements_*.txt files, below.

## Instructions

### Setup

1. Create new virtual environment
2. Install packages from requirements_txt_files/requirements_setup.txt.
3. Run `python setup.py`
4. Create roles for model trainer, endpoint creator, endpoint, and inference client. See sample policies
   for IAM in the IAM_policies directory.
5. Supply role names, bucket names, etc. in common/constants.py

### Train model

1. Create new virtual environment
2. Install packages from requirements_txt_files/requirements_training.txt.
3. Assume model-trainer role
4. Run `python -m training.start_training`

### Create endpoint

1. Create new virtual environment
2. Install packages from requirements_txt_files/requirements_endpoint.txt.
3. Assume endpoint-creator role
4. Run `python -m inference.endpoint.start_inference_endpoint`
5. Record enpoint name

### Run inference

1. Create new virtual environment
2. Install packages from requirements_txt_files/requirements_client.txt.
3. Assume inference client role
4. Run `python -m inference.client.run_inference <endpoint_name>`

### Clean up resources

1. Delete the SageMaker AI endpoint using the AWS CLI or console
   (`aws sagemaker delete-endpoint --endpoint-name <endpoint_name>``)
2. Delete the endpoint configuration
   (`aws sagemaker delete-endpoint-config --endpoint-config-name <config_name>``),
3. Delete the model (`aws sagemaker delete-model --model-name <model_name>`)
4. Delete S3 bucket contents and the bucket itself to avoid storage charges
5. Optionally delete Amazon CloudWatch log groups with /aws/sagemaker/ prefix.

Note that training jobs cannot be deleted and are retained indefinitely.


## License

This library is licensed under the MIT-0 License. See the LICENSE file.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
