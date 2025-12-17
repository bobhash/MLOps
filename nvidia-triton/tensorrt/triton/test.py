import base64

import numpy as np
import tritonclient.http as httpclient


def to_b64(path: str) -> str:
    with open(path, "rb") as f:
        jpeg_bytes = f.read()

    b64_bytes = base64.b64encode(jpeg_bytes)
    b64_str = b64_bytes.decode("utf-8")
    return b64_str


def call_triton(path: str):
    b64 = to_b64(path)
    client = httpclient.InferenceServerClient(url="localhost:8500")
    input_data = httpclient.InferInput("IMAGE_B64", [1], "BYTES")
    input_data.set_data_from_numpy(np.array([b64], dtype=object))
    output_data = httpclient.InferRequestedOutput("EMBEDDING_TRT", binary_data=True)
    response = client.infer(
        model_name="ensemble-embedders",
        inputs=[input_data],
        outputs=[
            httpclient.InferRequestedOutput("EMBEDDING_TRT", binary_data=True),
            httpclient.InferRequestedOutput("EMBEDDING_ONNX", binary_data=True),
        ],
    )
    print(f"{response.as_numpy('EMBEDDING_ONNX').shape=}")
    print(f"{response.as_numpy('EMBEDDING_TRT').shape=}")
    print(response.as_numpy("EMBEDDING_ONNX") - response.as_numpy("EMBEDDING_TRT"))


if __name__ == "__main__":
    call_triton(
        "/home/toomuch/mlops-course-2025-f/nvidia-triton/tensorrt/triton/img.jpeg"
    )
