# SocialSecurity

## Requirements

To install requirements:

```bash
conda create -n social python=3.10 -y
conda activate social

# Install Pytorch (Depend on your CUDA version, see https://pytorch.org/get-started/previous-versions/)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install -r requirements.txt
```