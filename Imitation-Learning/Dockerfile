FROM learningbydoingdocker/codalab-competition-docker:latest

# necessary for system-wide installation
USER root 
## Install pytorch-cpu inside it
RUN pip install torch==1.9.0+cpu  torchvision==0.10.0+cpu \ 
    torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html \
    pytorch-lightning