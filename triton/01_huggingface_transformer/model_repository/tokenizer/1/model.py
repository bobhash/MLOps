import numpy as np
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.tokenzier = transformers.AutoTokenizer.from_pretrained(
            "/assets/tokenizer", local_files_only=True
        )

    def tokenize(self, texts):
        encoded = self.tokenzier(
            texts, padding="max_length", max_length=16, truncation=True
        )
        input_ids = np.array(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.array(encoded["attention_mask"], dtype=np.int32)
        return input_ids, attention_mask

    def execute(self, requests):
        responses = []
        for request in requests:
            texts = pb_utils.get_input_tensor_by_name(request, "TEXTS").as_numpy()
            texts = [el.decode() for el in texts]

            input_ids, attention_mask = self.tokenize(texts)

            output_input_ids = pb_utils.Tensor("INPUT_IDS", input_ids)
            output_attention_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_input_ids, output_attention_mask]
            )
            responses.append(inference_response)

        return responses
