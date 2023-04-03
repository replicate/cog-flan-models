import json
import argparse
import os
import shutil
from typing import Optional
from config import HUGGINGFACE_MODEL_NAME, load_tokenizer

from transformers import T5ForConditionalGeneration
from torch.utils.data import Dataset
import torch.distributed as dist
import torch
from transformers import Trainer, TrainingArguments
from cog import Input, BaseModel, Path, File
from tensorizer import TensorSerializer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"

def reset_dir(directory):
    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)

class TrainingOutput(BaseModel):
    weights: Path

class DatasetBuilder:
    """Dataset agnostic class to take in input_ids and labels and spit out tokens"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def batch_tokenize(self, texts):
        """Tokenizes text. Presently doesn't pad inputs, just returns input ids."""
        tokenized = [
            self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
            ).input_ids
            for prompt in texts
        ]
        return tokenized

    def construct_dataset(self, input_data):
        prompts = [val['prompt'] for val in input_data]
        tokenized_input_ids = self.batch_tokenize(prompts)
        labels = [val["output"] for val in input_data]
        tokenized_labels = self.batch_tokenize(labels)
        return TuneDataset(tokenized_input_ids, tokenized_labels)


class TuneDataset(Dataset):
    """Dead simple torch dataset wrapper. Attention masks are created in collator"""

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class CustomDataCollatorSeq2Seq:
    """Collate examples for dynamic batch construction in supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple(
            [instance[key][0] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )  # -100 tells torch to ignore these tokens in loss computation.
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def load_json(path):
    """Loads a single json blob"""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_model(model_name_or_path):
    if model_name_or_path is None:
        model_name_or_path = HUGGINGFACE_MODEL_NAME
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path, cache_dir="pretrained_weights"
    )

    return model


def load_peft_model(model_name_or_path, lora_rank: int, lora_alpha: int, lora_dropout: float):
    if model_name_or_path is None:
        model_name_or_path = HUGGINGFACE_MODEL_NAME
    model = T5ForConditionalGeneration.from_pretrained(
        model_name_or_path, cache_dir="pretrained_weights", torch_dtype=torch.float16, load_in_8bit=True, device_map="auto"
    )    
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r = lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias='none',
        task_type=TaskType.SEQ_2_SEQ_LM
    )
    model = get_peft_model(model, config)
    return model


def train(
    train_data: Path = Input(description="path to data file to use for fine-tuning your model"),
    eval_data: Path = Input(description="path to optional evaluation data file to use for model eval", default=None),
    weights: Optional[Path] = Input(description="location of weights that are going to be fine-tuned", default=None),
    train_batch_size: int = Input(description="batch size per GPU", default=8, ge=1),
    gradient_accumulation_steps: int = Input(description="number of training steps to update gradient for before performing a backward pass", default=8),
    lr_scheduler_type: str = Input(description="learning rate scheduler", default="cosine", choices=["linear", "cosine", 'cosine_with_restarts', 'polynomial', 'inverse_sqrt', 'constant', 'constant_with_warmup']),
    learning_rate: float = Input(description="learning rate, for learning!", default=2e-4, ge=0),
    warmup_ratio: float = Input(description="pct of steps for a linear learning rate warmup", ge=0, le=0.5, default=0.03),
    num_train_epochs: int = Input(description="number of training epochs", ge=1, default=1),
    max_steps: int = Input(description="number of steps to run training for, supersedes num_train_epochs", default=None, ge=0),
    logging_steps: int = Input(description="number of steps between logging epoch & loss", default=1),
    peft: bool = Input(description="fine-tune model using lora w/peft", default=False),
    lora_rank: int = Input(description="if training lora, rank of lora matrices", default=8),
    lora_alpha: int = Input(description="alpha if training lora", default=8),
    lora_dropout: float = Input(description="dropout if training lora", default=0.1)
) -> TrainingOutput:  
    reset_dir(CHECKPOINT_DIR)
    if os.path.exists(MODEL_OUT):
        os.remove(MODEL_OUT)
    print("loading model")
    if peft:
        print("training lora!")
        model = load_peft_model(weights, lora_rank, lora_alpha, lora_dropout)
    else:
        model = load_model(weights)
    tokenizer = load_tokenizer()
    print("loading dataset")
    print(train_data)
    train_data = load_json(train_data)
    p = DatasetBuilder(tokenizer)
    train_dataset = p.construct_dataset(train_data)
    eval_dataset = None
    if eval_data:
        eval_data = load_json(eval_data)
        eval_dataset = p.construct_dataset(eval_data)
    print("training")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=TrainingArguments(
            tf32=True,
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy=SAVE_STRATEGY,
            save_total_limit=1,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            max_steps=max_steps,
            fsdp="full_shard auto_wrap"
        ),
        data_collator=CustomDataCollatorSeq2Seq(tokenizer),
    )
    trainer.train()
    # tensorize!
    model = trainer.model
    serializer = TensorSerializer(MODEL_OUT)
    serializer.write_module(model)
    serializer.close()
    
    path_out = Path(MODEL_OUT)
    to_return = TrainingOutput(weights=path_out)
    return to_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on a text dataset"
    )

    parser.add_argument(
        "--train_data", type=Path, required=True, help="Path to the json dataset"
    )
    parser.add_argument(
        "--eval_data", type=Path, required=False, help="Path to the json dataset", default=None
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="The model class to fine-tune on HF or as a local path (e.g. 'google/flan-t5-xxl'",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Number of warmup steps for the learning rate scheduler",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--peft",
        action="store_true"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.4,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default='cosine',
    )
    some_args = parser.parse_args()
    train(**vars(some_args))

