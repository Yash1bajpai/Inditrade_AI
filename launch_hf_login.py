import subprocess

ssh_command = """
cd /teamspace/studios/this_studio/Inditrade_AI
echo '#!/bin/bash' > start.sh
echo 'export PATH="/home/zeus/miniconda3/envs/cloudspace/bin:$PATH"' >> start.sh
echo 'pip install -r src/models/requirements_finetune.txt' >> start.sh
echo 'huggingface-cli login --token "yhf_vArijLPmrimRTvIKRdxPvNAvznrxXqiHbO"' >> start.sh
echo 'python src/models/llm_qlora.py --epochs 3 > finetune.log 2>&1' >> start.sh
chmod +x start.sh
tmux kill-session -t finetune 2>/dev/null
tmux new-session -d -s finetune bash ./start.sh
"""

process = subprocess.run(
    ["ssh", "s_01kxd886566pdvqqacq2bmwzrg@ssh.lightning.ai", ssh_command],
    capture_output=True,
    text=True
)
