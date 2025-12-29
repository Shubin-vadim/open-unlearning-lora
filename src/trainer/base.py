# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
import torch
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any
from utils.logging import get_logger

logger = get_logger(__name__)


class FinetuneTrainer(Trainer):
    def __init__(self, evaluators=None, template_args=None, *args, **kwargs):
        self.evaluators = evaluators
        self.template_args = template_args
        super().__init__(*args, **kwargs)

    def _move_model_to_device(self, model, device):
        """
        Override to handle meta tensors and device_map properly.
        When a model is loaded with device_map, it's already placed on devices
        and we shouldn't try to move it. Also handles meta tensor errors.
        """
        # Check if model was loaded with device_map (has hf_device_map attribute)
        # This works for both regular models and PEFT-wrapped models
        has_device_map = False
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            has_device_map = True
        elif hasattr(model, "base_model") and hasattr(model.base_model, "hf_device_map") and model.base_model.hf_device_map:
            has_device_map = True
        elif hasattr(model, "model") and hasattr(model.model, "hf_device_map") and model.model.hf_device_map:
            has_device_map = True
        
        if has_device_map:
            logger.debug(
                "Model was loaded with device_map, skipping device movement."
            )
            return model

        # Check if model has any parameters on meta device
        has_meta_params = False
        try:
            for param in model.parameters():
                if hasattr(param, "device") and param.device.type == "meta":
                    has_meta_params = True
                    break
        except Exception:
            # If we can't check parameters, try the default behavior
            pass

        if has_meta_params:
            logger.warning(
                "Model contains meta tensors. Model should be loaded with proper device_map or initialized weights."
            )
            # Don't try to move meta tensors - they should be handled during model loading
            return model

        # Use the default behavior for non-meta tensors
        try:
            model = model.to(device)
        except NotImplementedError as e:
            if "meta tensor" in str(e).lower() or "meta" in str(e).lower():
                logger.warning(
                    "Caught meta tensor error. Model may need to be loaded with proper device_map or weights initialized."
                )
                # Check again if model has device_map (might be nested)
                has_device_map = False
                if hasattr(model, "hf_device_map") and model.hf_device_map:
                    has_device_map = True
                elif hasattr(model, "base_model") and hasattr(model.base_model, "hf_device_map") and model.base_model.hf_device_map:
                    has_device_map = True
                elif hasattr(model, "model") and hasattr(model.model, "hf_device_map") and model.model.hf_device_map:
                    has_device_map = True
                
                if has_device_map:
                    logger.debug("Found device_map on nested model, returning as-is.")
                    return model
                
                # Otherwise, this is an error condition
                raise RuntimeError(
                    "Model contains meta tensors that cannot be moved. "
                    "Please ensure the model is loaded with proper weights or device_map configuration."
                ) from e
            else:
                raise

        return model

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        trial: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        # Run a custom evaluator and save results
        if self.evaluators:
            logger.info(f"Running custom evaluators: {list(self.evaluators.keys())}")
            if self.accelerator.is_local_main_process:
                eval_metrics = {}
                if self.accelerator.num_processes == 1:
                    run_dir = self._get_output_dir(trial=trial)
                    checkpoint_folder = (
                        f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                    )
                    output_dir = os.path.join(run_dir, checkpoint_folder, "evals")
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Saving evaluation results to: {output_dir}")
                    eval_metrics = {}
                    for evaluator_name, evaluator in self.evaluators.items():
                        logger.info(f"Running evaluator: {evaluator_name}")
                        eval_args = {
                            "output_dir": output_dir,
                            "template_args": self.template_args,
                            "model": self.model,
                            "tokenizer": self.tokenizer,
                        }
                        metrics = evaluator.evaluate(**eval_args)
                        eval_metrics.update(metrics)
                        logger.info(f"Evaluator {evaluator_name} completed with {len(metrics)} metrics")
                    self.log(eval_metrics)
                    logger.info(f"Evaluation completed. Total metrics: {len(eval_metrics)}")
                else:
                    logger.warning(
                        "Custom evaluator can be run with this Trainer only when a single accelerator process is running."
                    )
                return eval_metrics

        if eval_dataset is None:
            logger.debug("No eval_dataset provided, returning empty metrics")
            return {}
        # Run the default HF Trainer evaluate method when eval dataset is provided
        logger.info(f"Running default HF Trainer evaluation on eval_dataset with {len(eval_dataset)} samples")
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
