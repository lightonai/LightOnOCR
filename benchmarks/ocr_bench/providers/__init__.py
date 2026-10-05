"""ParseBench providers and pipelines not in upstream, registered on import.

* ``loocr_grounding``: pipelines ``loocr_grounding_parse_with_layout``, ``loocr_grounding_omit_margins``,
  ``loocr_grounding_omit_strict`` and ``loocr_grounding_raw``.
* ``mistral_ocr_hf``: pipeline ``mistral_ocr_4_1_no_hf``.
"""

from parse_bench.evaluation.layout_adapters.adapters import QwenLayoutAdapter
from parse_bench.extensions import register_layout_adapter, register_pipeline
from parse_bench.inference.pipelines import get_pipeline
from parse_bench.schemas.pipeline import PipelineSpec
from parse_bench.schemas.product import ProductType

from ocr_bench.providers import (  # noqa: F401  (registers the providers)
    loocr_grounding,
    mistral_ocr_hf,
)

# 400 DPI capped at 5M pixels per page (instead of QwenProvider's 150 DPI): best of an olmOCR-bench
# DPI x max-pixels sweep (2026-10-02, 3 repeats), tied with 600 DPI / 7M within noise.
register_pipeline(
    PipelineSpec(
        pipeline_name="loocr_grounding_parse_with_layout",
        provider_name="loocr_grounding",
        product_type=ProductType.PARSE,
        config={
            "model": "loocr-grounding",
            "server_url_env": "LOOCR_SERVER_URL",
            "prompt": "grounding",
            "prompt_mode": "parse_with_layout",
            "dpi": 400,
            "max_pixels": 5_000_000,
            "temperature": 0.1,
            # QwenProvider's 16384 default is the whole context: image + output tokens would overflow.
            "max_tokens": 12288,
        },
    )
)

# Same inference; on olmOCR-bench and fr-bench the markdown differs (RAW_MARKDOWN in ocr_bench/runner.py):
# * omit_margins drops page margins and side notes (LoocrGroundingProvider.omit_margins).
# * omit_strict also drops footnotes, keeps side notes (LoocrGroundingProvider.omit_strict).
# * raw is the model output as is, block markers included.
for _variant in ("omit_margins", "omit_strict", "raw"):
    register_pipeline(
        PipelineSpec(
            pipeline_name=f"loocr_grounding_{_variant}",
            provider_name="loocr_grounding",
            product_type=ProductType.PARSE,
            config=get_pipeline("loocr_grounding_parse_with_layout").config,
        )
    )

# Upstream mistral_ocr_4_1 plus extract_header / extract_footer.
register_pipeline(
    PipelineSpec(
        pipeline_name="mistral_ocr_4_1_no_hf",
        provider_name="mistral_ocr_hf",
        product_type=get_pipeline("mistral_ocr_4_1").product_type,
        config={**get_pipeline("mistral_ocr_4_1").config, "extract_header": True, "extract_footer": True},
    )
)


@register_layout_adapter("loocr_grounding", priority=90)
class LoocrGroundingLayoutAdapter(QwenLayoutAdapter):
    """The provider emits QwenProvider's layout_pages shape, so Qwen's adapter applies as is."""

    @classmethod
    def matches(cls, inference_result) -> bool:
        return False  # resolved by provider key only; never claim other providers' results
