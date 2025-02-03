import boto3
import time
import base64
import argparse
from botocore.exceptions import ClientError

def launch_training_instance(image_uri, region):
    """Launch an EC2 instance and run training in Docker container."""
    ec2 = boto3.client('ec2', region_name=region)
    
    # Create user data script with extensive error handling
    user_data = f"""#!/bin/bash
exec > >(tee /var/log/user-data.log) 2>&1
set -x

echo "=== Starting initialization at $(date) ==="

# Function to log with timestamp
log() {{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}}

# Error handling
handle_error() {{
    log "ERROR: $1"
    log "Saving system logs..."
    dmesg > /var/log/dmesg.log
    journalctl -xe > /var/log/journal.log
    log "Error occurred. Instance will continue running for debugging."
    exit 1
}}

trap 'handle_error "Script failed"' ERR

# System update
log "Updating system packages..."
yum update -y || handle_error "System update failed"

# Install Docker
log "Installing Docker..."
yum install -y docker || handle_error "Docker installation failed"
log "Starting Docker service..."
systemctl start docker || handle_error "Docker service start failed"
systemctl status docker
docker --version || handle_error "Docker not properly installed"

# Check AWS connectivity
log "Testing AWS connectivity..."
aws sts get-caller-identity || handle_error "AWS credentials not working"

# Login to ECR
log "Logging into ECR..."
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {image_uri.split('/')[0]} || handle_error "ECR login failed"

# Pull image
log "Pulling Docker image..."
docker pull {image_uri} || handle_error "Failed to pull Docker image"

# Create directories
log "Creating mount directories..."
mkdir -p /opt/ml/model /opt/ml/input/data/training

# Run container
log "Starting training container..."
docker run --rm \
    -v /opt/ml:/opt/ml \
    -e AWS_DEFAULT_REGION={region} \
    -e WANDB_PROJECT="WindFarm_Curriculum" \
    {image_uri} train || handle_error "Container failed to start"

log "Training complete"
shutdown -h now
"""
    
    try:
        print("Launching EC2 instance...")
        response = ec2.run_instances(
            ImageId='ami-0c7217cdde317cfec',  # Amazon Linux 2
            InstanceType='t3.medium',
            MinCount=1,
            MaxCount=1,
            UserData=base64.b64encode(user_data.encode()).decode(),
            IamInstanceProfile={
                'Name': 'ec2-sagemaker-role'
            },
            SecurityGroups=['default'],
            TagSpecifications=[
                {
                    'ResourceType': 'instance',
                    'Tags': [
                        {
                            'Key': 'Name',
                            'Value': 'WindFarm-Training'
                        }
                    ]
                }
            ]
        )
        
        instance_id = response['Instances'][0]['InstanceId']
        print(f"\nInstance ID: {instance_id}")
        print("\nInitialization process:")
        print("1. System update and Docker installation (2-3 minutes)")
        print("2. ECR login and image pull (1-2 minutes)")
        print("3. Container startup (1 minute)")
        print("\nWaiting for instance to start...")
        
        # Wait for instance to be running
        waiter = ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id])
        
        return instance_id
        
    except ClientError as e:
        print(f"Error launching instance: {str(e)}")
        return None

def get_instance_logs(instance_id, region):
    """Get various logs from the instance."""
    ec2 = boto3.client('ec2', region_name=region)
    try:
        response = ec2.get_console_output(InstanceId=instance_id)
        return response.get('Output', '')
    except:
        return "Unable to fetch logs"

def monitor_instance(instance_id, region):
    """Monitor the EC2 instance with improved error detection."""
    ec2 = boto3.client('ec2', region_name=region)
    last_state = None
    start_time = time.time()
    
    print("\nMonitoring instance output:")
    print("=" * 50)
    print(f"Instance ID: {instance_id}")
    print(f"To terminate: aws ec2 terminate-instances --instance-ids {instance_id}")
    print("=" * 50 + "\n")
    
    try:
        while True:
            try:
                # Get instance details
                response = ec2.describe_instances(InstanceIds=[instance_id])
                instance = response['Reservations'][0]['Instances'][0]
                state = instance.get('State', {}).get('Name')
                
                # Print state changes
                if state != last_state:
                    print(f"\nInstance state changed to: {state}")
                    last_state = state
                
                # Handle different states
                if state == 'running':
                    logs = get_instance_logs(instance_id, region)
                    print(logs)
                    time.sleep(10)
                elif state in ['stopped', 'terminated']:
                    print("\nInstance stopped or terminated.")
                    print("\nFinal logs:")
                    print(get_instance_logs(instance_id, region))
                    break
                elif state == 'pending':
                    print(".", end='', flush=True)
                    time.sleep(5)
                else:
                    print(f"\nUnexpected state: {state}")
                    time.sleep(5)
                
            except Exception as e:
                print(f"\nError monitoring instance: {str(e)}")
                time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Instance is still running.")
        print(f"To terminate instance: aws ec2 terminate-instances --instance-ids {instance_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-uri', required=True, help='ECR image URI')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    args = parser.parse_args()
    
    instance_id = launch_training_instance(args.image_uri, args.region)
    if instance_id:
        monitor_instance(instance_id, args.region)
    else:
        print("Failed to launch training instance")

if __name__ == '__main__':
    main()
