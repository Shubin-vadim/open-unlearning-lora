import logging
from typing import Any, Dict

import torch
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer

from trainer.base import FinetuneTrainer
from utils.logging import get_logger

logger = get_logger(__name__)

TRAINER_REGISTRY: Dict[str, Any] = {}


def _register_trainer(trainer_class):
    TRAINER_REGISTRY[trainer_class.__name__] = trainer_class


def _lazy_import_unlearn_trainers():
    """Lazy import of unlearn trainers."""
    from trainer.unlearn.grad_ascent import GradAscent
    from trainer.unlearn.grad_diff import GradDiff
    from trainer.unlearn.npo import NPO
    from trainer.unlearn.dpo import DPO
    from trainer.unlearn.simnpo import SimNPO
    from trainer.unlearn.rmu import RMU
    from trainer.unlearn.undial import UNDIAL
    from trainer.unlearn.ceu import CEU
    from trainer.unlearn.satimp import SatImp
    from trainer.unlearn.wga import WGA
    from trainer.unlearn.pdu import PDU
    
    # Register all unlearn trainers
    _register_trainer(GradAscent)
    _register_trainer(GradDiff)
    _register_trainer(NPO)
    _register_trainer(DPO)
    _register_trainer(SimNPO)
    _register_trainer(RMU)
    _register_trainer(UNDIAL)
    _register_trainer(CEU)
    _register_trainer(SatImp)
    _register_trainer(WGA)
    _register_trainer(PDU)


def load_trainer_args(trainer_args: DictConfig, dataset):
    trainer_args = dict(trainer_args)
    warmup_epochs = trainer_args.pop("warmup_epochs", None)
    if warmup_epochs:
        batch_size = trainer_args["per_device_train_batch_size"]
        grad_accum_steps = trainer_args["gradient_accumulation_steps"]
        num_devices = torch.cuda.device_count() or 1  # Default to 1 if no CUDA devices
        dataset_len = len(dataset) if dataset is not None else 0
        
        if dataset_len == 0:
            logger.warning("Dataset is empty, setting warmup_steps to 0")
            warmup_steps = 0
        else:
            denominator = batch_size * grad_accum_steps * num_devices
            if denominator == 0:
                logger.warning(f"Denominator is zero (batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, num_devices={num_devices}), setting warmup_steps to 0")
                warmup_steps = 0
            else:
                warmup_steps = int(
                    (warmup_epochs * dataset_len)
                    // denominator
                )
        trainer_args["warmup_steps"] = warmup_steps
        logger.info(f"Calculated warmup_steps={warmup_steps} from warmup_epochs={warmup_epochs}, dataset_len={dataset_len}, batch_size={batch_size}, grad_accum_steps={grad_accum_steps}, num_devices={num_devices}")

    logger.debug(f"Training arguments: {trainer_args}")
    trainer_args = TrainingArguments(**trainer_args)
    logger.info(f"Initialized TrainingArguments with output_dir={trainer_args.output_dir}")
    return trainer_args


def load_trainer(
    trainer_cfg: DictConfig,
    model,
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    evaluators=None,
    template_args=None,
):
    trainer_args = trainer_cfg.args
    method_args = trainer_cfg.get("method_args", {})
    logger.info(f"Loading trainer with handler: {trainer_cfg.get('handler')}")
    if method_args:
        logger.debug(f"Method-specific args: {method_args}")
    trainer_args = load_trainer_args(trainer_args, train_dataset)
    trainer_handler_name = trainer_cfg.get("handler")
    assert trainer_handler_name is not None, ValueError(
        f"{trainer_handler_name} handler not set"
    )
    trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    # If trainer not found, try lazy loading unlearn trainers
    if trainer_cls is None:
        logger.debug(f"Trainer '{trainer_handler_name}' not found in registry, attempting lazy import")
        _lazy_import_unlearn_trainers()
        trainer_cls = TRAINER_REGISTRY.get(trainer_handler_name, None)
    assert trainer_cls is not None, NotImplementedError(
        f"{trainer_handler_name} not implemented or not registered"
    )
    logger.info(f"Initializing {trainer_handler_name} trainer")
    train_size = len(train_dataset) if train_dataset is not None else 0
    eval_size = len(eval_dataset) if eval_dataset is not None else 0
    logger.debug(f"Train dataset size: {train_size}, Eval dataset size: {eval_size}")
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=trainer_args,
        evaluators=evaluators,
        template_args=template_args,
        **method_args,
    )
    logger.info(
        f"{trainer_handler_name} Trainer loaded, output_dir: {trainer_args.output_dir}"
    )
    return trainer, trainer_args


# Register Finetuning Trainer
_register_trainer(Trainer)
_register_trainer(FinetuneTrainer)

# Unlearn trainers are registered lazily via _lazy_import_unlearn_trainers()
