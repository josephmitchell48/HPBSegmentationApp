#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git libglib2.0-0 libsm6 libxrender1 libxext6

python3 -m venv /opt/hpb-venv
source /opt/hpb-venv/bin/activate

pip install --upgrade pip
pip install -r /opt/hpb-seg/requirements.txt

# Create model directories
sudo mkdir -p /models/nnunet_v1 /models/totalseg
sudo chown -R $USER:$USER /models

cat <<'EON' > /opt/hpb-seg/.env
RESULTS_FOLDER=/models/nnunet_v1
HPB_IN_ROOT=/tmp/hpb_in
HPB_OUT_ROOT=/tmp/hpb_out
EON

echo "Bootstrap complete. Place nnU-Net weights under /models/nnunet_v1." 
