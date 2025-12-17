# TensorRT

Как уже было сказано, папйлайн компиляции любой нейросетевой модели в TensorRT выглядит так:
1. Сварить ONNX из pytorch модели (`torch2onnx.py`)
2. При необходимости вызвать autocast для перевода нужных слоев в FP16
3. При необходимости вызвать quantize
4. Вызвать trtexec с флагом stronglyTypedNetwork.


# Step by step

### Step 1: Clone HF model
```bash
cd ./huggingface
git-lfs clone https://huggingface.co/google/siglip2-base-patch32-256
```

### Step 2: Export to ONNX
```bash
python3 torch2onnx.py \
--model_path ./huggingface/siglip2-base-patch32-256 \
--onnx_savepath ./ONNX/siglip2-base-patch32-256.onnx
```

### Step 3: Prepare calib data
```bash
python3 prepare-calib.py \
--num_imgs 32 \
--preprocessor_path ./huggingface/siglip2-base-patch32-256 \
--batch_size 8 \
--calib_images_path calib-images \
--savepath ./calib-cache/32_calib_base.npz
```

### Step 4: Run FP16 AutoCast 
```bash
python -m modelopt.onnx.autocast \
--onnx_path ./ONNX/siglip2-base-patch32-256.onnx \
--output_path ./ONNX/siglip2-base-patch32-256.fp16-autocast.onnx \
--low_precision_type fp16 \
--keep_io_types \
--calibration_data ./calib-cache/32_calib_base.npz \
--providers cpu \
--low_precision_type fp16 \
--log_level DEBUG \
> autocast-log.txt 2>&1
```


### Step 5: Compare original ONNX with autocasted
```bash
python3 compare-onnxs.py \
--fp32_onnx ./ONNX/siglip2-base-patch32-256.onnx \
--converted_onnx ./ONNX/siglip2-base-patch32-256.fp16-autocast.onnx \
--images_dir calib-images \
--num_images_to_test 64 \
--batch_size 16 \
--preprocessor_path ./huggingface/siglip2-base-patch32-256 \
--ort_execution_provider CPUExecutionProvider
```


### Step 6: Quantize to INT8
```bash
python -m modelopt.onnx.quantization \
--onnx_path ./ONNX/siglip2-base-patch32-256.onnx \
--output_path ./ONNX/siglip2-base-patch32-256.int8.onnx \
--quantize_mode int8 \
--high_precision_dtype fp16 \
--calibration_data_path ./calib-cache/32_calib_base.npz \
--calibration_shapes IMAGES:1x3x256x256 \
--calibration_eps cpu \
--passes concat_elimination \
--op_types_to_exclude LayerNormalization Softmax Div Sqrt Pow Exp ReduceMean Sub Conv Mul Add \
--op_types_to_exclude_fp16 LayerNormalization Softmax Div Sqrt Pow Exp ReduceMean Sub Conv Mul Add \
--calibration_method max \
--op_types_to_quantize Gemm MatMul \
--disable_mha_qdq \
--nodes_to_exclude /vision_model/embeddings/patch_embedding/Conv \
--log_level DEBUG \
> quantize-log.txt 2>&1
```


### Step 7: Compare original ONNX with Quantized
```bash
python3 compare-onnxs.py \
--fp32_onnx ./ONNX/siglip2-base-patch32-256.onnx \
--converted_onnx ./ONNX/siglip2-base-patch32-256.int8.onnx \
--images_dir calib-images \
--num_images_to_test 64 \
--batch_size 16 \
--preprocessor_path ./huggingface/siglip2-base-patch32-256 \
--ort_execution_provider CPUExecutionProvider
```

### Step 8: Compile TRT
```bash
docker run -it --rm --gpus '"device=2"' -v ./:/models nvcr.io/nvidia/tensorrt:25.08-py3
docker run -it --rm --gpus '"device=2"' -v ./:/models nvcr.io/nvidia/tritonserver:25.08-py3
docker run -it --rm --gpus '"device=2"' -v ./:/models nvcr.io/nvidia/tritonserver:25.08-py3-sdk


# Original (FP32)
trtexec \
--onnx=/models/ONNX/siglip2-base-patch32-256.onnx \
--saveEngine=/models/TRT/siglip2-base-patch32-256.plan \
--builderOptimizationLevel=5 \
--minShapes=IMAGES:1x3x256x256 \
--optShapes=IMAGES:4x3x256x256 \
--maxShapes=IMAGES:4x3x256x256 \
--verbose \
--profilingVerbosity=detailed \
--dumpProfile \
--separateProfileRun \
--stronglyTyped \
--noCompilationCache \
> /models/trt-log.txt 2>&1

# Autocast (FP16)
trtexec \
--onnx=/models/ONNX/siglip2-base-patch32-256.fp16-autocast.onnx \
--saveEngine=/models/TRT/siglip2-base-patch32-256.fp16-autocast.plan \
--builderOptimizationLevel=5 \
--minShapes=IMAGES:1x3x256x256 \
--optShapes=IMAGES:4x3x256x256 \
--maxShapes=IMAGES:4x3x256x256 \
--verbose \
--profilingVerbosity=detailed \
--dumpProfile \
--separateProfileRun \
--stronglyTyped \
--noCompilationCache

# Autocast (FP16) AMPERE+
trtexec \
--onnx=/models/ONNX/siglip2-base-patch32-256.fp16-autocast.onnx \
--saveEngine=/models/TRT/siglip2-base-patch32-256.fp16-autocast.ampere+.plan \
--builderOptimizationLevel=5 \
--minShapes=IMAGES:1x3x256x256 \
--optShapes=IMAGES:4x3x256x256 \
--maxShapes=IMAGES:4x3x256x256 \
--verbose \
--profilingVerbosity=detailed \
--dumpProfile \
--separateProfileRun \
--stronglyTyped \
--noCompilationCache \
--hardwareCompatibilityLevel=ampere+

# Autocast (FP16) Separate Profile
trtexec \
--onnx=/models/ONNX/siglip2-base-patch32-256.fp16-autocast.onnx \
--saveEngine=/models/TRT/siglip2-base-patch32-256.fp16-autocast.multiprofile.plan \
--verbose \
--profilingVerbosity=detailed \
--dumpProfile \
--separateProfileRun \
--stronglyTyped \
--builderOptimizationLevel=5 \
--profile=0 \
--minShapes=IMAGES:1x3x256x256 \
--optShapes=IMAGES:1x3x256x256 \
--maxShapes=IMAGES:1x3x256x256 \
--profile=1 \
--minShapes=IMAGES:2x3x256x256 \
--optShapes=IMAGES:2x3x256x256 \
--maxShapes=IMAGES:2x3x256x256 \
--profile=2 \
--minShapes=IMAGES:3x3x256x256 \
--optShapes=IMAGES:3x3x256x256 \
--maxShapes=IMAGES:3x3x256x256 \
--profile=3 \
--minShapes=IMAGES:4x3x256x256 \
--optShapes=IMAGES:4x3x256x256 \
--maxShapes=IMAGES:4x3x256x256
```


### Step 9: Measure pure TRT performance

```bash
docker run -it --rm --gpus '"device=2"' -v ./:/models nvcr.io/nvidia/tensorrt:25.08-py3

# Pure FP32 (Original)
trtexec --loadEngine=/models/TRT/siglip2-base-patch32-256.plan \
--shapes=IMAGES:4x3x256x256 \
--avgRuns=64 \
--iterations=128 \
--warmUp=32 \
--streams=1 \
--useCudaGraph \
--useSpinWait \
--noDataTransfers

# AutoCast FP16
trtexec --loadEngine=/models/TRT/siglip2-base-patch32-256.fp16-autocast.plan \
--shapes=IMAGES:4x3x256x256 \
--avgRuns=64 \
--iterations=128 \
--warmUp=32 \
--streams=1 \
--useCudaGraph \
--useSpinWait \
--noDataTransfers

# AutoCast FP16 AMPERE+
trtexec --loadEngine=/models/TRT/siglip2-base-patch32-256.fp16-autocast.ampere+.plan \
--shapes=IMAGES:4x3x256x256 \
--avgRuns=64 \
--iterations=128 \
--warmUp=32 \
--streams=1 \
--useCudaGraph \
--useSpinWait \
--noDataTransfers
```
