ARG VARIANT="latest-gpu-jupyter"
FROM tensorflow/tensorflow:${VARIANT}

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libcudnn9-cuda-12  -y 

# [Optional] Uncomment if you want to install an additional version
#  of node using nvm
# ARG EXTRA_NODE_VERSION=18
# RUN su node -c "source /usr/local/share/nvm/nvm.sh \
#    && nvm install ${EXTRA_NODE_VERSION}"
