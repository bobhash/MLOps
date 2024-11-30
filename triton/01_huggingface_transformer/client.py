import numpy as np

from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from tritonclient.utils import np_to_triton_dtype


def call_triton_ensemble(text: str):
    client = InferenceServerClient(url="0.0.0.0:8500")
    text = np.array([text.encode("utf-8")], dtype=object)

    input_text = InferInput(
        name="TEXTS", shape=text.shape, datatype=np_to_triton_dtype(text.dtype)
    )
    input_text.set_data_from_numpy(text, binary_data=True)

    response = client.infer(
        "ensemble",
        [input_text],
        outputs=[
            InferRequestedOutput("EMBEDDINGS-ONNX"),
            InferRequestedOutput("EMBEDDINGS-TRT"),
        ],
    )
    return (
        response.as_numpy("EMBEDDINGS-ONNX"),
        response.as_numpy("EMBEDDINGS-TRT"),
    )


def main():
    text = "bla bla bla"
    outputs_onnx, outputs_trt = call_triton_ensemble(text)
    print(f"{outputs_onnx=}")
    print(f"{outputs_trt=}")


if __name__ == "__main__":
    main()
