from transformers import AutoConfig, LlamaForCausalLM
import torch
import torch.nn as nn
import logging
import gc
from copy import deepcopy
from transformers import AutoModelForCausalLM
from utils.logging import get_logger

logger = get_logger("model")


class ProbedLlamaForCausalLM(LlamaForCausalLM):
    """
    Class for loading a LlamaForCausalLM model with the following custom behavior:
    - Initializes only the first `n_layers` of the model.
    - Sets up a newly initialized `lm_head`, optionally using weights from
     `head_pretrained_model_name_or_path`
    - Trains only the lm_head parameters with rest of the model frozen.
    - Once the model is saved during training, for inference it can also be loaded using
      AutoModelForCausalLM
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        head_pretrained_model_name_or_path: str = None,
        n_layers: int = 100,
        freeze_base_model: bool = True,
        **kwargs,
    ):
        logger.info(f"Initializing ProbedLlamaForCausalLM from {pretrained_model_name_or_path}")
        logger.debug(f"Parameters: n_layers={n_layers}, freeze_base_model={freeze_base_model}, head_pretrained_model_name_or_path={head_pretrained_model_name_or_path}")
        
        logger.debug("Loading model configuration...")
        config, unused_kwargs = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
        )
        original_num_layers = config.num_hidden_layers
        config.tie_word_embeddings = False
        logger.debug(f"Original model has {original_num_layers} layers, will use {n_layers} layers")
        
        logger.info("Loading base model...")
        model: LlamaForCausalLM = super().from_pretrained(
            pretrained_model_name_or_path, config=config, **unused_kwargs
        )
        logger.info("Base model loaded successfully")

        # Limit number of transformer layers
        n_layers = min(n_layers, model.config.num_hidden_layers)
        if n_layers < original_num_layers:
            logger.info(f"Limiting model to {n_layers} layers (from {original_num_layers} total layers)")
        model.config.num_hidden_layers = n_layers
        model.model.layers = nn.ModuleList(model.model.layers[:n_layers])
        logger.debug(f"Model layers truncated to {len(model.model.layers)} layers")

        # Reinitialize lm_head
        ref_params = list(model.model.layers[-1].parameters())[0]
        device = ref_params.device
        logger.debug(f"Using device: {device} for lm_head")
        if head_pretrained_model_name_or_path is not None:
            logger.info(
                f"Initialising lm_head from {head_pretrained_model_name_or_path}"
            )
            logger.debug("Loading head model...")
            head_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
                head_pretrained_model_name_or_path, config=config, **unused_kwargs
            )
            logger.debug("Copying lm_head weights...")
            lm_head = deepcopy(head_model.lm_head).to(device)
            model.set_output_embeddings(lm_head)
            logger.info("lm_head initialized from pretrained model")
        else:
            logger.info("Initialising new lm_head with random weights")
            model._init_weights(model.lm_head)
            logger.debug("lm_head weights initialized")

        # Cleanup
        logger.debug("Performing cleanup (gc.collect, torch.cuda.empty_cache)...")
        gc.collect()
        torch.cuda.empty_cache()

        # Set trainable params
        logger.debug(f"Setting trainable parameters (freeze_base_model={freeze_base_model})...")
        trainable_params = []
        frozen_params = []
        for name, p in model.named_parameters():
            is_trainable = not freeze_base_model or name.startswith("lm_head")
            p.requires_grad = is_trainable
            if is_trainable:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        logger.info(f"Trainable parameters: {len(trainable_params)} (lm_head only: {freeze_base_model})")
        logger.debug(f"Trainable param names: {trainable_params[:5]}{'...' if len(trainable_params) > 5 else ''}")
        logger.info(
            f"Initialised a ProbedLlamaForCausalLM model with {n_layers} layers"
        )
        return model
