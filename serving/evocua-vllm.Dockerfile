FROM vllm/vllm-openai@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a

# EvoCUA's frozen model card recommends Transformers 4.57.3.  The upstream
# vLLM 0.11.0 image contains 4.57.0, so keep the vLLM/CUDA stack intact and
# only pin this model-specific Python dependency.
RUN python3 -m pip install --no-cache-dir --upgrade "transformers==4.57.3"
