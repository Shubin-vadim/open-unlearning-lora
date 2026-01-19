"""
Attack implementations.
"""

from transformers import AutoModelForCausalLM
import torch

from evals.metrics.base import unlearning_metric
from evals.metrics.mia.loss import LOSSAttack
from evals.metrics.mia.min_k import MinKProbAttack
from evals.metrics.mia.min_k_plus_plus import MinKPlusPlusAttack
from evals.metrics.mia.gradnorm import GradNormAttack
from evals.metrics.mia.zlib import ZLIBAttack
from evals.metrics.mia.reference import ReferenceAttack

from evals.metrics.mia.utils import mia_auc
import logging

logger = logging.getLogger("metrics")

## NOTE: all MIA attack statistics are signed as required in order to show the
# same trends as loss (higher the score on an example, less likely the membership)


@unlearning_metric(name="mia_loss")
def mia_loss(model, **kwargs):
    result = mia_auc(
        LOSSAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
    )
    if result is None:
        # GPU OOM - return a placeholder result
        return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
    return result


@unlearning_metric(name="mia_min_k")
def mia_min_k(model, **kwargs):
    result = mia_auc(
        MinKProbAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )
    if result is None:
        # GPU OOM - return a placeholder result
        return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
    return result


@unlearning_metric(name="mia_min_k_plus_plus")
def mia_min_k_plus_plus(model, **kwargs):
    result = mia_auc(
        MinKPlusPlusAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )
    if result is None:
        # GPU OOM - return a placeholder result
        return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
    return result


@unlearning_metric(name="mia_gradnorm")
def mia_gradnorm(model, **kwargs):
    result = mia_auc(
        GradNormAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        p=kwargs["p"],
    )
    if result is None:
        # GPU OOM - return a placeholder result
        return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
    return result


@unlearning_metric(name="mia_zlib")
def mia_zlib(model, **kwargs):
    result = mia_auc(
        ZLIBAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        tokenizer=kwargs.get("tokenizer"),
    )
    if result is None:
        # GPU OOM - return a placeholder result
        return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
    return result


@unlearning_metric(name="mia_reference")
def mia_reference(model, **kwargs):
    if "reference_model_path" not in kwargs:
        raise ValueError("Reference model must be provided in kwargs")
    try:
        logger.info(f"Loading reference model from {kwargs['reference_model_path']}")
        reference_model = AutoModelForCausalLM.from_pretrained(
            kwargs["reference_model_path"],
            torch_dtype=model.dtype,
            device_map={"": model.device},
        )
        result = mia_auc(
            ReferenceAttack,
            model,
            data=kwargs["data"],
            collator=kwargs["collators"],
            batch_size=kwargs["batch_size"],
            reference_model=reference_model,
        )
        if result is None:
            # GPU OOM - return a placeholder result
            return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
        return result
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or ("cuda" in error_str and "memory" in error_str):
            logger.warning(
                f"GPU out of memory error during mia_reference. "
                f"Skipping this metric. Error: {e}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"agg_value": None, "auc": None, "error": "GPU out of memory"}
        else:
            raise
