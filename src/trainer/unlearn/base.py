from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from packaging import version
from trainer.base import FinetuneTrainer
from utils.logging import get_logger

logger = get_logger(__name__)

from transformers.trainer_pt_utils import (
    nested_detach,
)


from transformers.utils import (
    is_sagemaker_mp_enabled,
)


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import (
        smp_forward_only,
        smp_nested_concat,
    )
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


class UnlearnTrainer(FinetuneTrainer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable gradient checkpointing on base model for PEFT/LoRA models
        # This is necessary because PEFT models need gradient checkpointing enabled
        # on the base model, not just the wrapper
        if self.args.gradient_checkpointing:
            self._enable_gradient_checkpointing_for_peft()
    
    def _enable_gradient_checkpointing_for_peft(self):
        """Enable gradient checkpointing on base model for PEFT/LoRA models."""
        try:
            # Check if model is a PEFT model
            if hasattr(self.model, 'base_model'):
                # Disable use_cache when gradient checkpointing is enabled
                # This is required for compatibility with gradient checkpointing
                if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False
                    logger.info("Disabled use_cache for gradient checkpointing compatibility")
                
                # Enable input require grads for PEFT models
                # This ensures that inputs can participate in the gradient computation graph
                try:
                    if hasattr(self.model, 'enable_input_require_grads'):
                        self.model.enable_input_require_grads()
                        logger.info("Enabled input require grads for PEFT/LoRA model")
                except Exception as e:
                    logger.debug(f"Could not enable input require grads (may not be available): {e}")
                
                base_model = self.model.base_model
                # Check if base_model has a model attribute (for nested structures)
                if hasattr(base_model, 'model'):
                    base_model = base_model.model
                
                # Disable use_cache on base model config as well
                if hasattr(base_model, 'config') and hasattr(base_model.config, 'use_cache'):
                    base_model.config.use_cache = False
                
                # Enable gradient checkpointing on the base model
                if hasattr(base_model, 'gradient_checkpointing_enable'):
                    base_model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing on base model for PEFT/LoRA")
                elif hasattr(base_model, 'config'):
                    # For some models, we need to enable it via config
                    if hasattr(base_model.config, 'gradient_checkpointing'):
                        base_model.config.gradient_checkpointing = True
                        logger.info("Enabled gradient checkpointing via config for PEFT/LoRA")
                
                # Verify that trainable parameters exist
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                if trainable_params == 0:
                    logger.warning("No trainable parameters found! Gradient checkpointing may not work correctly.")
                else:
                    logger.debug(f"Found {trainable_params:,} trainable parameters for gradient checkpointing")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing for PEFT model: {e}")
            # Don't fail if we can't enable it - let the Trainer handle it

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        The only change to this function is calling the Trainer's compute_loss, as it's often overridden by unlearning methods, and we want to maintain the Trainer's evaluation setup.
        """
        logger.debug(f"Starting prediction step: prediction_loss_only={prediction_loss_only}")
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )
        logger.debug(f"Prediction step config: has_labels={has_labels}, return_loss={return_loss}, loss_without_labels={loss_without_labels}")

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []
        logger.debug(f"Ignore keys: {ignore_keys}")

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                logger.debug("Using SageMaker Model Parallel for prediction")
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    logger.debug("Computing loss using super().compute_loss() for evaluation")
                    with self.compute_loss_context_manager():
                        ### Call compute_loss of super class since overridden compute_loss is not be applicable to eval_dataset.
                        loss, outputs = super().compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()
                    logger.debug(f"Computed loss: {loss.item() if loss is not None else None}")

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            logger.debug("Returning only loss (prediction_loss_only=True)")
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        
        logits_shape = logits.shape if hasattr(logits, 'shape') else 'N/A'
        logger.debug(f"Prediction step completed: loss={loss.item() if loss is not None else None}, logits_shape={logits_shape}")

        return (loss, logits, labels)
