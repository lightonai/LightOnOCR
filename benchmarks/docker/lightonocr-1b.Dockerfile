# LightOnOCR-1B checkpoints use the Mistral3/Pixtral architecture. vLLM v0.30.0's pixtral.py imports
# PixtralRotaryEmbedding, which transformers 5.17 (the image's version) renamed: pin 5.16.1.
FROM vllm/vllm-openai:v0.30.0
RUN pip install --no-cache-dir "transformers==5.16.1"
