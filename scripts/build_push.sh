#!/bin/bash
set -e

# Get AWS account ID from current credentials
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REPOSITORY_NAME="windfarm-ppo"

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names ${REPOSITORY_NAME} || \
    aws ecr create-repository --repository-name ${REPOSITORY_NAME}

# Get the login command from ECR and execute it directly
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build the docker image
docker build -t ${REPOSITORY_NAME} .

# Tag the image
docker tag ${REPOSITORY_NAME}:latest ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:latest

# Push the image
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPOSITORY_NAME}:latest
