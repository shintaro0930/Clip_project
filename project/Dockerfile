ARG nvidia_cuda_version=11.4.0-cudnn8-devel-ubuntu20.04

FROM nvidia/cuda:${nvidia_cuda_version}
 
RUN apt-get update && \
    apt-get install -y python3 python3-pip

# PyTorchとCLIPのインストール
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ftfy regex tqdm && \
    pip3 install clip && \
    pip3 install matplotlib && \
    pip3 install Image

# RUN pip install ftfy regex tqdm
# RUN pip install git+https://github.com/openai/CLIP.git
