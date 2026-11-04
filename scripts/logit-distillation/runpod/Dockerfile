FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build
ARG CUDA_ARCHITECTURES=80
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
RUN --mount=type=cache,target=/opt/aikar-engine/build-runpod ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && cmake -S . -B build-runpod -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DGGML_NATIVE=OFF -DGGML_CUDA_FA=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_SERVER=OFF && cmake --build build-runpod --target llama-extract-logits llama-finetune-qlion test-backend-ops test-logit-distillation -j "${BUILD_JOBS}" && mkdir -p /opt/runtime && cp -a build-runpod/bin/. /opt/runtime/

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 libssl3 time ripgrep ca-certificates && python3 -m pip install --no-cache-dir numpy==2.2.6 tqdm pyyaml requests && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/aikar-engine
COPY --from=build /opt/runtime /opt/aikar-engine/bin
COPY gguf-py /opt/aikar-engine/gguf-py
COPY scripts/logit-distillation /opt/aikar-engine/scripts/logit-distillation
ENV REPO=/opt/aikar-engine BIN_DIR=/opt/aikar-engine/bin MODEL_DIR=/workspace/Lumen-3.8-Pulsar DATA_KO=/workspace/train_ko.jsonl DATA_OTHER=/workspace/train.jsonl GPU_LAYERS=99 STATIC_GRAPH=off CTX=512 EXTRACT_UBATCH=512 TRAIN_UBATCH=512 THREADS=8 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=/opt/aikar-engine/bin
ENTRYPOINT ["bash", "/opt/aikar-engine/scripts/logit-distillation/runpod/entrypoint.sh"]
CMD ["help"]
