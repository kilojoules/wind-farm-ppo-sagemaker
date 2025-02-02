# scripts/setup_env.sh
#!/bin/bash
set -e

# Clone DTUWindGym if it doesn't exist
if [ ! -d "DTUWindGym" ]; then
    echo "Cloning DTUWindGym..."
    mkdir -p DTUWindGym
    git clone git@gitlab.windenergy.dtu.dk:path/to/DTUWindGym.git DTUWindGym
    cd DTUWindGym
    pip install -e .
    cd ..
fi
