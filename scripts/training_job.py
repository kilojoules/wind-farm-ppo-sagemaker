# scripts/training_job.py
import os
import boto3
from sagemaker.pytorch import PyTorch
from dotenv import load_dotenv

load_dotenv()

account_id = os.getenv('AWS_ACCOUNT_ID')
region = os.getenv('AWS_DEFAULT_REGION')
bucket = f'windfarm-ppo-{account_id}'

estimator = PyTorch(
    entry_point='train_sagemaker.py',
    source_dir='src',
    image_uri=f'{account_id}.dkr.ecr.{region}.amazonaws.com/windfarm-ppo:latest',
    role=f'arn:aws:iam::{account_id}:role/SageMakerRole',
    instance_count=1,
    instance_type='ml.g4dn.xlarge',
    hyperparameters={
        'dt_env': 1,
        'power_avg': 30,
        'lookback_window': 100,
        'learning_rate': 1e-5
    },
    use_spot_instances=True,
    max_wait=3600*8
)

estimator.fit({'training': f's3://{bucket}/input'})
