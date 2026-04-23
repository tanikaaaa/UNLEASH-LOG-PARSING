from collections import Counter
import json
import sys
import time
sys.path.append('..')
import os

from unleash.models.roberta import RobertaForLogParsing
from unleash.models.deberta import DebertaForLogParsing
from unleash.data.data_loader import DataLoaderForPromptTuning
from unleash.tuning.trainer import Trainer, TrainingArguments
from unleash.parsing_base import template_extraction

from transformers import set_seed
import logging
from unleash.arguments import get_args
import pandas as pd
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger("unleash")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger.handlers[0].setFormatter(formatter)


if __name__ == "__main__":
    data_args, model_args, train_args, common_args = get_args()

    if common_args.seed is not None:
        set_seed(common_args.seed)

    assert data_args.train_file is not None, "A training file is required"

    # Load model
    if model_args.model_name_or_path == "microsoft/deberta-base":
        p_model = DebertaForLogParsing(model_args.model_name_or_path, ct_loss_weight=0.1)
    else:
        p_model = RobertaForLogParsing(model_args.model_name_or_path, ct_loss_weight=0.1)

    # Load data
    data_loader = DataLoaderForPromptTuning(data_args)

    logger.debug(f"{data_args.dataset_name} loaded with {len(data_loader.raw_datasets['train'])} train samples")
    logger.debug(f"Text column: {data_loader.text_column_name} | Label column: {data_loader.label_column_name}")

    p_model.tokenizer = data_loader.initialize(p_model.tokenizer)
    data_loader.tokenize()

    data_loader.build_dataloaders(
        train_args.per_device_train_batch_size,
        train_args.per_device_eval_batch_size
    )

    # Training setup
    training_args = TrainingArguments(
        output_dir=common_args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        weight_decay=0.0,
        num_train_epochs=train_args.num_train_epochs,
        max_train_steps=train_args.max_train_steps,
        gradient_accumulation_steps=1,
        lr_scheduler_type="polynomial"
    )

    trainer = Trainer(
        model=p_model,
        args=training_args,
        train_loader=data_loader.get_train_dataloader(),
        eval_loader=data_loader.get_val_dataloader(),
        compute_metrics=None,
        no_train_samples=len(data_loader.raw_datasets['train']),
        device=device,
    )

    # Train model
    t0 = time.time()
    p_model = trainer.train()

    if common_args.save_model:
        trainer.save_pretrained(f"{common_args.output_dir}/models/{data_args.dataset_name}")

    # =========================
    # 🔥 LOG PARSING (FIXED)
    # =========================

    
    # Load raw log file (instead of CSV)
    print("USING LOG FILE:", data_args.log_file)
    with open(data_args.log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    log_df = pd.DataFrame({
        "Content": lines
    })

    logs = log_df["Content"].tolist()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    t1 = time.time()
    p_model.eval()

    devices = [device] * common_args.parsing_num_processes

    templates, model_time, model_invocations = template_extraction(
        p_model, devices, logs, vtoken=data_loader.vtoken
    )

    log_df["EventTemplate"] = pd.Series(templates)

    # =========================
    # SAVE RESULTS
    # =========================

    t2 = time.time()

    task_output_dir = f"{common_args.output_dir}/logs"
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)

    log_df.to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_structured.csv",
        index=False
    )

    # Save templates
    counter = Counter(templates)
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    template_df = pd.DataFrame(items, columns=["EventTemplate", "Occurrence"])
    template_df["EventID"] = [f"E{i+1}" for i in range(len(template_df))]

    template_df[["EventID", "EventTemplate", "Occurrence"]].to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_templates.csv",
        index=False
    )

    # Save time stats
    time_cost_file = f"{task_output_dir}/time_cost.json"

    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, "r") as f:
            time_table = json.load(f)

    time_table[data_args.dataset_name] = {
        "TrainingTime": round(t1 - t0, 3),
        "ParsingTime": round(t2 - t1, 3),
        "ModelTime": round(model_time, 3),
        "ModelInvocations": model_invocations
    }

    with open(time_cost_file, "w") as f:
        json.dump(time_table, f)

    print("\n✅ UNLEASH pipeline completed successfully.\n")