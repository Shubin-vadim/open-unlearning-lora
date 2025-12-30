# import torch
from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loading CompletionDataset with hf_args: {hf_args}")
        
        # Handle MUSE dataset split naming issues
        hf_args = hf_args.copy()
        path = hf_args.get("path", "")
        split = hf_args.get("split")
        
        # Handle muse-bench/MUSE datasets: map 'retain' to 'retain1'
        if split == "retain" and "muse-bench/MUSE" in path:
            logger.warning(
                "MUSE dataset does not have a 'retain' split. "
                "Available splits are: 'retain1', 'retain2', 'forget', 'holdout'. "
                "Using 'retain1' as fallback. To use both retain splits, use 'retain1+retain2'."
            )
            hf_args["split"] = "retain1"
        
        # Handle tamarsonha/MUSE datasets: map 'retain1'/'retain2' to 'retain'
        elif split in ("retain1", "retain2") and "tamarsonha/MUSE" in path:
            logger.info(
                f"Mapping split '{split}' to 'retain' for tamarsonha/MUSE dataset. "
                "Available splits are: 'full', 'retain'."
            )
            hf_args["split"] = "retain"
        
        try:
            self.data = load_hf_dataset(**hf_args)
        except ValueError as e:
            if "Unknown split" in str(e):
                error_msg = str(e)
                # Try to extract available splits from error message
                if "Should be one of" in error_msg:
                    logger.error(
                        f"Invalid split '{hf_args.get('split')}' for dataset '{path}'. {error_msg}"
                    )
                elif "retain" in error_msg.lower():
                    if "tamarsonha/MUSE" in path:
                        logger.error(
                            f"Invalid split '{split}' for tamarsonha/MUSE dataset. "
                            "Available splits are: 'full', 'retain'. "
                            "Note: 'retain1' and 'retain2' are automatically mapped to 'retain'."
                        )
                    else:
                        logger.error(
                            f"Invalid split '{hf_args.get('split')}' for MUSE dataset. "
                            "Available splits: 'retain1', 'retain2', 'forget', 'holdout'. "
                            "To use both retain splits, use 'retain1+retain2'."
                        )
                else:
                    logger.error(f"Invalid split '{hf_args.get('split')}' for dataset '{path}'. {error_msg}")
            raise
        
        self.data = add_dataset_index(self.data)
        logger.info(f"Loaded {len(self.data)} completion samples")
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space
        logger.debug(f"CompletionDataset initialized: prefix_key='{prefix_key}', text_key='{text_key}', max_length={max_length}")

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        logger.info(f"Loading PretrainingDataset with hf_args: {hf_args}")
        
        # Handle MUSE dataset split naming issues
        hf_args = hf_args.copy()
        path = hf_args.get("path", "")
        split = hf_args.get("split")
        
        # Handle muse-bench/MUSE datasets: map 'retain' to 'retain1'
        if split == "retain" and "muse-bench/MUSE" in path:
            logger.warning(
                "MUSE dataset does not have a 'retain' split. "
                "Available splits are: 'retain1', 'retain2', 'forget', 'holdout'. "
                "Using 'retain1' as fallback. To use both retain splits, use 'retain1+retain2'."
            )
            hf_args["split"] = "retain1"
        
        # Handle tamarsonha/MUSE datasets: map 'retain1'/'retain2' to 'retain'
        elif split in ("retain1", "retain2") and "tamarsonha/MUSE" in path:
            logger.info(
                f"Mapping split '{split}' to 'retain' for tamarsonha/MUSE dataset. "
                "Available splits are: 'full', 'retain'."
            )
            hf_args["split"] = "retain"
        
        try:
            raw_data = load_hf_dataset(**hf_args)
        except ValueError as e:
            if "Unknown split" in str(e):
                error_msg = str(e)
                # Try to extract available splits from error message
                if "Should be one of" in error_msg:
                    logger.error(
                        f"Invalid split '{hf_args.get('split')}' for dataset '{path}'. {error_msg}"
                    )
                elif "retain" in error_msg.lower():
                    if "tamarsonha/MUSE" in path:
                        logger.error(
                            f"Invalid split '{split}' for tamarsonha/MUSE dataset. "
                            "Available splits are: 'full', 'retain'. "
                            "Note: 'retain1' and 'retain2' are automatically mapped to 'retain'."
                        )
                    else:
                        logger.error(
                            f"Invalid split '{hf_args.get('split')}' for MUSE dataset. "
                            "Available splits: 'retain1', 'retain2', 'forget', 'holdout'. "
                            "To use both retain splits, use 'retain1+retain2'."
                        )
                else:
                    logger.error(f"Invalid split '{hf_args.get('split')}' for dataset '{path}'. {error_msg}")
            raise
        
        logger.info(f"Loaded raw data with {len(raw_data[text_key])} text entries")
        self.chunks = self._chunk_raw_text(raw_data[text_key])
        logger.info(f"Created {len(self.chunks)} chunks from raw text (max_length={max_length})")

    def _chunk_raw_text(self, raw_text):
        raw_text = "\n\n".join(raw_text)
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)[
            "input_ids"
        ]
        num_chunks = len(full_token_sequence) // self.max_length + 1
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                self.tokenizer.decode(
                    full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
                )
            )
        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return preprocess_pretraining_instance(
            self.tokenizer, "", self.chunks[idx], self.max_length
        )
