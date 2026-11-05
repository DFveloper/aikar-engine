FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build
ARG CUDA_ARCHITECTURES=80;120
ARG BUILD_JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git libssl-dev python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/aikar-engine
COPY CMakeLists.txt LICENSE AUTHORS ./
COPY cmake ./cmake
COPY common ./common
COPY include ./include
COPY src ./src
COPY ggml ./ggml
COPY vendor ./vendor
COPY licenses ./licenses
COPY examples ./examples
COPY tests ./tests
COPY tools ./tools
COPY pocs ./pocs
COPY app ./app
RUN --mount=type=cache,id=aikar-cuda128,target=/opt/aikar-engine/build-runpod ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S . -B build-runpod -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGGML_NATIVE=OFF -DGGML_CUDA_FA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build build-runpod --target llama-extract-logits llama-finetune-qlion test-backend-ops test-logit-distillation -j "${BUILD_JOBS}" && mkdir -p /opt/runtime && cp -a build-runpod/bin/. /opt/runtime/

RUN /usr/local/cuda/bin/cuobjdump --list-elf /opt/runtime/libggml-cuda.so > /opt/runtime/cuda-architectures.txt

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS cuda-runtime

FROM runpod/base:1.0.7-ubuntu2404@sha256:7e45ffb69c0625b22c3fe90886b3192e95e0b9ea44d0861333a874ce37a07187
COPY --from=cuda-runtime /usr/local/cuda-12.8 /usr/local/cuda-12.8
COPY --from=cuda-runtime /usr/lib/x86_64-linux-gnu/libnccl.so.2* /usr/local/lib/
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libssl3t64 time ripgrep && python -m pip install --no-cache-dir numpy==2.2.6 tqdm pyyaml requests huggingface_hub==1.22.0 pyarrow==20.0.0 && rm -rf /var/lib/apt/lists/* && ln -s /usr/local/cuda-12.8 /usr/local/cuda && ldconfig
WORKDIR /opt/aikar-engine
COPY --from=build /opt/runtime /opt/aikar-engine/bin
COPY gguf-py /opt/aikar-engine/gguf-py
COPY scripts/logit-distillation /opt/aikar-engine/scripts/logit-distillation
ENV REPO=/opt/aikar-engine BIN_DIR=/opt/aikar-engine/bin MODEL_DIR=/workspace/Lumen-3.8-Pulsar DATA_KO=/workspace/train_ko.jsonl DATA_OTHER=/workspace/train.jsonl GPU_LAYERS=99 STATIC_GRAPH=off CTX=1024 EXTRACT_UBATCH=1024 TRAIN_UBATCH=1024 THREADS=8 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/opt/aikar-engine/bin:/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN chmod +x scripts/logit-distillation/runpod/entrypoint.sh && ln -s /opt/aikar-engine/scripts/logit-distillation/runpod/entrypoint.sh /usr/local/bin/distill && mv /start.sh /runpod-start.sh && cp scripts/logit-distillation/runpod/start-pod.sh /start.sh && chmod +x /start.sh
EXPOSE 8888 22
ENTRYPOINT []
CMD ["/start.sh"]
