FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04


RUN apt update \
    && apt install -y --no-install-recommends wget python3 python3-pip \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools


ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get  -y -f install build-essential gcc cmake nvidia-container-toolkit nvidia-container-runtime git

# ADD Fastllm
ARG FUNCTION_DIR="/opt/program"
WORKDIR ${FUNCTION_DIR}
RUN git clone https://github.com/ztxz16/fastllm.git ${FUNCTION_DIR}/fastllm
# ADD ./fastllm/ ${FUNCTION_DIR}/fastllm

# Install fastllm
WORKDIR ${FUNCTION_DIR}
RUN mkdir -p ${FUNCTION_DIR}/fastllm/build
RUN cd fastllm/build && cmake .. -DUSE_CUDA=ON && make -j && cd tools && python3 setup.py install

#ADD Python file
ADD requirements.txt ${FUNCTION_DIR}
RUN pip3 install -r ${FUNCTION_DIR}/requirements.txt
RUN pip3 install flask gevent

#ADD model
RUN mkdir -p ${FUNCTION_DIR}/model
# ADD chatglm2_fastllm_model.flm ${FUNCTION_DIR}/model/model.flm
ADD s5cmd ${FUNCTION_DIR}/
ENV MODEL_PATH="${FUNCTION_DIR}/model/"

ADD handler.py ${FUNCTION_DIR}
ADD sm_llm.py ${FUNCTION_DIR}


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PYTHONIOENCODING="utf8"
ENV PATH="/opt/ml/:${PATH}"

WORKDIR ${FUNCTION_DIR}
ENTRYPOINT ["python3", "sm_llm.py"]