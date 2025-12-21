# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import os
import logging
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
