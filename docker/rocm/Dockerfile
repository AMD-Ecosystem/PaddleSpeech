# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# PaddleSpeech on AMD Instinct GPUs (ROCm)
#
# This image runs PaddleSpeech unchanged on top of the ROCm build of PaddlePaddle.
# PaddleSpeech has no CUDA/HIP kernels of its own: all GPU compute is dispatched by
# PaddlePaddle via paddle.set_device('gpu'), which routes to the AMD GPU. The only
# difference from the CUDA image is the framework wheel (ROCm PaddlePaddle instead
# of paddlepaddle-gpu) and the base image (a ROCm base instead of an NVIDIA CUDA base).
#
# Build:
#   # The ROCm PaddlePaddle wheel is provided as a build argument (a local wheel path
#   # placed in the build context, or a pip-resolvable name/index once published).
#   docker build -t paddlespeech:rocm \
#     --build-arg PADDLE_ROCM_WHEEL=paddlepaddle_dcu-3.4.0.dev20260825-cp311-cp311-linux_x86_64.whl \
#     -f docker/rocm/Dockerfile .
#
# Run (GPU passthrough -- no nvidia-docker / extra container toolkit needed):
#   docker run --rm -it \
#     --device=/dev/kfd --device=/dev/dri --group-add video \
#     --security-opt seccomp=unconfined --ipc=host --shm-size=8g \
#     paddlespeech:rocm bash
#
# ROCm base image: the manylinux2.28 (RHEL 8.10) ROCm image. Pick the current tag for
# your ROCm stream; this default is validated for ROCm 10.1.
ARG ROCM_BASE=registry.aifoundry.amd.com/rocm-manylinux228-rhel810:10.1.0a20260821
FROM ${ROCM_BASE}

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:/opt/rocm/lib:${LD_LIBRARY_PATH}

# The ROCm PaddlePaddle wheel filename. Default is the wheel delivered on the
# AMD-AIOSS/hipSHIFT port/Paddle branch (https://github.com/AMD-AIOSS/hipSHIFT/tree/port/Paddle);
# download it into the build context before building, or override this ARG with the
# filename/path of the wheel you placed in the context. Do NOT install
# paddlepaddle-gpu here -- that is the CUDA wheel.
ARG PADDLE_ROCM_WHEEL=paddlepaddle_dcu-3.4.0.dev20260825-cp311-cp311-linux_x86_64.whl

WORKDIR /opt/paddlespeech
COPY . /opt/paddlespeech

RUN python3 -m ensurepip --upgrade >/dev/null 2>&1 || true && \
    python3 -m pip install --upgrade pip && \
    # PADDLE_ROCM_WHEEL is the wheel file in the build context (or an absolute path to it).
    python3 -m pip install "${PADDLE_ROCM_WHEEL}" && \
    python3 -c "import paddle; print('Paddle ROCm build:', paddle.is_compiled_with_rocm())"

# Build deps for PaddleSpeech's CPU C-extension dependencies (webrtcvad, pyworld,
# ligo-segments): the manylinux python needs its dev headers, plus Cython + numpy<2.
# Pin numpy<2 -- the ROCm PaddlePaddle wheel and paddleaudio are built against the
# numpy 1.x ABI, so letting a dep drag numpy 2.x raises a module-ABI RuntimeWarning.
RUN (command -v dnf >/dev/null 2>&1 && dnf install -y python3.11-devel || true) && \
    python3 -m pip install --upgrade "setuptools>=61" wheel pytest-runner cython "numpy<2"

# PaddleSpeech + its CPU C++ feature-extraction extensions (kaldi-native-fbank, sox).
# These are framework-agnostic CPU code and are reused unchanged. Use
# --no-build-isolation so the build backend runs in the main interpreter (the
# manylinux image ships a protobuf nspkg .pth that breaks pip's isolation subprocess).
RUN python3 -m pip install --no-build-isolation . && \
    python3 -m pip install "numpy<2" && \
    (python3 -m pip install paddlespeech_ctcdecoders || true) && \
    python3 -c "import paddlespeech; from paddlespeech.cli.asr.infer import ASRExecutor; print('PaddleSpeech import OK')" && \
    rm -rf ~/.cache/pip

# Verify the AMD GPU is visible to PaddlePaddle at build time (best-effort; the GPU is
# only present at `docker run` time with the --device passthrough above).
CMD ["bash"]
