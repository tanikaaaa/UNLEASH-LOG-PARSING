import re
import torch
from transformers import DataCollatorForTokenClassification
import os
import pandas as pd


DOWNLOAD_URL = "https://zenodo.org/records/8275861/files/{}.zip"

def generate_logformat_regex(log_format):
    import re

    headers = []
    splitters = re.split(r'(<[^<>]+>)', log_format)

    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(r' +', r'\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<>').strip()
            regex += f'(?P<{header}>.*?)'
            headers.append(header)

    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers, size=None):
    """ 
    Function to transform log file to contents
    Args:
        log_file: log file path
        regex: regular expression to split log messages
        headers: headers of log messages
        size: number of log messages to read
    Returns:
        log_messages: list of log contents
    """
    log_contents = []
    with open(log_file, 'r') as file:
        if size is None:
            log_lines = file.readlines()
        else:
            log_lines = [next(file) for _ in range(size)]
        for line in log_lines:
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_contents.append(message[-1])
            except Exception as e:
                pass
    return log_contents


class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name]
                  for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels']
                      for feature in features] if 'ori_labels' in features[0].keys() else None
        max_length = max([len(x['input_ids']) for x in features])
        # print(max_length)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=min(max_length, 256),
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch
        # print(batch['labels'])
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] *
                               (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
            # batch['ori_labels'] = None
        else:
            batch["labels"] = [[self.label_pad_token_id] *
                               (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]
            # batch['ori_labels'] = None
        # print(batch)
        batch = {k: torch.tensor(v, dtype=torch.int64)
                 for k, v in batch.items()}
        return batch

def load_loghub_dataset(dataset_name="Apache", cache_dir=None, format="csv", log_format=None):
    """
    Load dataset from local data folder (NO DOWNLOAD)
    """

    # point directly to your local data folder
    dataset_dir = os.path.join("..", "data", dataset_name)

    if format == "csv":
        log_df = pd.read_csv(f"{dataset_dir}/{dataset_name}_full.log_structured.csv")

    elif format == "text":
        file_path = f"{dataset_dir}/{dataset_name}_full.log"

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        log_df = pd.DataFrame({
            "Content": lines,
            "EventTemplate": [""] * len(lines)
        })

    return log_df