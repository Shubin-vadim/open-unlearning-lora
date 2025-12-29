import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from evals.metrics.utils import (
    aggregate_to_1D,
    evaluate_probability,
    eval_text_similarity,
    run_batchwise_evals,
    tokenwise_vocab_logprobs,
)
from evals.metrics.base import unlearning_metric

# Supress the info messages logged while calculating rouge using rouge_scorer
logging.getLogger("absl").setLevel(logging.WARNING)
logger = logging.getLogger("evaluator")


@unlearning_metric(name="probability")
def probability(model, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    prob_values = torch.tensor(
        [
            evals["prob"]
            for evals in scores_by_index.values()
            if evals["prob"] is not None
        ],
        device=device
    )
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": float(torch.mean(prob_values).item()), "value_by_index": scores_by_index}


@unlearning_metric(name="probability_w_options")
def probability_w_options(model, **kwargs):
    """Normalize probabilities of correct answers against false answers for
    open-ended datasets, returning the aggregated value and per-index probabilities."""
    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    correct = torch.tensor(
        [correct_answer_results[idx]["prob"] for idx in filtered_indices],
        device=device
    )
    all_wrong = torch.tensor(
        [wrong_answer_results[idx]["prob"] for idx in filtered_indices],
        device=device
    )
    # Sum along all dimensions except the first (batch dimension)
    # If already 1D, this will sum over empty dims which returns the tensor as-is
    if all_wrong.ndim > 1:
        wrong = torch.sum(all_wrong, dim=tuple(range(1, all_wrong.ndim)))
    else:
        wrong = all_wrong
    probs = correct / (correct + wrong + 1e-10)

    # Convert to numpy only at the end for return values
    probs_np = probs.cpu().numpy()
    # Create a mapping from index to position in filtered_indices for efficient lookup
    filtered_idx_map = {idx: pos for pos, idx in enumerate(filtered_indices)}
    # Create value_by_index with all indices, but only include computed values for filtered indices
    value_by_index = {}
    for idx in correct_indices:
        if idx in filtered_idx_map:
            # Get the position in filtered_indices to access the corresponding prob value
            filtered_pos = filtered_idx_map[idx]
            value_by_index[idx] = {"prob": probs_np[filtered_pos]}
        else:
            # Keep None for indices that were filtered out
            value_by_index[idx] = {"prob": None}
    return {"agg_value": float(torch.mean(probs).item()), "value_by_index": value_by_index}


@unlearning_metric(name="rouge")
def rouge(model, **kwargs):
    """Calculate ROUGE metrics and return the aggregated value along with per-index scores."""
    tokenizer = kwargs["tokenizer"]
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    generation_args = kwargs["generation_args"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {"tokenizer": tokenizer, "generation_args": generation_args}
    scores_by_index = run_batchwise_evals(
        model,
        dataloader,
        eval_text_similarity,
        fun_args,
        "Calculating text similarity",
    )
    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    rouge_values = torch.tensor(
        [
            evals[kwargs["rouge_type"]]
            for evals in scores_by_index.values()
            if evals[kwargs["rouge_type"]] is not None
        ],
        device=device
    )
    rouge_values = aggregate_to_1D(rouge_values)
    return {
        "agg_value": float(torch.mean(rouge_values).item()),
        "value_by_index": scores_by_index,
    }


@unlearning_metric(name="truth_ratio")
def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""

    # Forget data: It is better if false and true are equally likely,
    # i.e., tr=false/true is closest to 1.
    def closer_to_1_better(arr):
        if isinstance(arr, torch.Tensor):
            return torch.mean(torch.minimum(arr, 1 / (arr + 1e-10))).item()
        else:
            return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    # Non-forget data: It is better if tr=false/true is lower, i.e.,
    # 1-tr is higher.
    def true_better(arr):
        if isinstance(arr, torch.Tensor):
            return torch.mean(torch.clamp(1 - arr, min=0)).item()
        else:
            return np.mean(np.maximum(0, 1 - arr))

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct_avg_losses = [
        correct_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]
    wrong_avg_losses = [
        wrong_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]

    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    correct_avg_losses = torch.tensor(correct_avg_losses, device=device)
    wrong_avg_losses = torch.tensor(wrong_avg_losses, device=device)
    
    correct_avg_losses = aggregate_to_1D(correct_avg_losses)
    wrong_avg_losses = aggregate_to_1D(wrong_avg_losses)

    correct_prob = torch.exp(-correct_avg_losses)
    wrong_prob = torch.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / (correct_prob + 1e-10)
    
    # Keep computations on GPU for aggregation
    truth_ratio_stats = truth_ratios
    forget_tr_avg = aggregator(truth_ratio_stats)
    
    # Convert to numpy only at the end for return values
    truth_ratios_np = truth_ratios.cpu().numpy()
    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios_np])
    )
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}


@unlearning_metric(name="exact_memorization")
def exact_memorization(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                logger.warning(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                # Keep computations on GPU
                preds = torch.argmax(log_probs, dim=-1)
                valid_len_tensor = torch.tensor(valid_len, device=log_probs.device, dtype=torch.float32)
                em_score = (preds == labels).sum().float() / valid_len_tensor
                # Convert to Python float only at the end
                em_batch.append({"score": em_score.item()})
        return em_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, fun_args, "Calculating EM"
    )
    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    em_values = torch.tensor(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ],
        device=device
    )
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": float(torch.mean(em_values).item()), "value_by_index": scores_by_index}


@unlearning_metric(name="extraction_strength")
def extraction_strength(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                logger.warning(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                # Keep computations on GPU
                device = log_probs.device
                preds = torch.argmax(log_probs, dim=-1)
                valid_len_tensor = torch.tensor(valid_len, device=device, dtype=torch.float32)
                k = valid_len  # Default value if no match found
                for k_idx in range(valid_len):
                    suff_preds = preds[k_idx:]
                    suff_labels = labels[k_idx:]
                    if torch.equal(suff_preds, suff_labels):
                        k = k_idx
                        break
                # Keep computation on GPU
                k_tensor = torch.tensor(k, device=device, dtype=torch.float32)
                es_score = 1 - (k_tensor / valid_len_tensor)
                # Convert to Python float only at the end
                es_batch.append({"score": es_score.item()})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    # Use torch tensors on GPU instead of numpy arrays
    device = next(model.parameters()).device
    es_values = torch.tensor(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ],
        device=device
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": float(torch.mean(es_values).item()), "value_by_index": scores_by_index}
