import os
import boto3
from sagemaker.pytorch import PyTorch

# Get AWS account ID
sts = boto3.client('sts')
ACCOUNT_ID = sts.get_caller_identity()['Account']

# Get region
session = boto3.session.Session()
REGION = session.region_name

estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='src',
    image_uri=f'{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/windfarm-ppo:latest',
    role=f'arn:aws:iam::{ACCOUNT_ID}:role/SageMakerRole',
    instance_count=1,
    instance_type='ml.t3.medium',
    hyperparameters={
        'dt_env': 1,
        'power_avg': 30,
        'lookback_window': 100,
        'learning_rate': 1e-5
    },
    use_spot_instances=True,
    max_wait=3600*8,
    environment={
        'WANDB_PROJECT': 'WindFarm_Curriculum'
    }
)

estimator.fit()
