# The Infinity-Parser2 checkpoint's tokenizer needs transformers 5.
FROM vllm/vllm-openai:v0.17.1
RUN pip install --no-cache-dir "transformers==5.3.0"
