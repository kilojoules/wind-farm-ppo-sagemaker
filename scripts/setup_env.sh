# scripts/setup_env.sh
#!/bin/bash
set -e

# Clone DTUWindGym if it doesn't exist
if [ ! -d "DTUWindGym" ]; then
    echo "Cloning DTUWindGym..."
    git clone git@gitlab.windenergy.dtu.dk:manils/dtu_wind_gym.git
    cd dtu_wind_gym
    pip install -e .
    cd ..
fi
