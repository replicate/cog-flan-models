import argparse
import os
from subprocess import call
from typing import Optional

import torch
from cog import BaseModel, Input, Path
from tensorizer import TensorSerializer
from transformers import T5ForConditionalGeneration

from config import HUGGINGFACE_MODEL_NAME

MODEL_OUT = "/src/tuned_weights.tensors"
CHECKPOINT_DIR = "checkpoints"
SAVE_STRATEGY = "epoch"
DIST_OUT_DIR = "tmp/model"


class TrainingOutput(BaseModel):
    weights: Path


def train(
    train_data: Path = Input(
        description="path to data file to use for fine-tuning your model"
    ),
    eval_data: Path = Input(
        description="path to optional evaluation data file to use for model eval",
        default=None,
    ),
    weights: Path = Input(
        description="location of weights that are going to be fine-tuned", default=None
    ),
    train_batch_size: int = Input(description="batch size per GPU", default=4, ge=1),
    gradient_accumulation_steps: int = Input(
        description="number of training steps to update gradient for before performing a backward pass",
        default=1,
    ),
    learning_rate: float = Input(
        description="learning rate, for learning!", default=2e-5, ge=0
    ),
    warmup_ratio: float = Input(
        description="pct of steps for a linear learning rate warmup",
        ge=0,
        le=0.5,
        default=0.03,
    ),
    num_train_epochs: int = Input(
        description="number of training epochs", ge=1, default=1
    ),
    max_steps: int = Input(
        description="number of steps to run training for, supersedes num_train_epochs",
        default=-1
    ),
    logging_steps: int = Input(
        description="number of steps between logging epoch & loss", default=1
    ),
    gradient_checkpointing: bool = Input(
        description="whether to use gradient checkpointing to save memory at the cost of speed",
        default=True
    ),
) -> TrainingOutput:
    input_model = weights if weights is not None else HUGGINGFACE_MODEL_NAME

    root_path = os.getcwd()
    deepspeed_config = os.path.join(root_path, "ds_config/ds_flan_t5_z3_config_bf16_no_offload.json")

    output_dir = DIST_OUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    num_gpus_flag = f"--num_gpus={num_gpus}"

    print(f"Local Output Dir: {output_dir}")
    print(f"Number of GPUs: {num_gpus}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_DATASETS_CACHE"] = "/src/.hf-cache"

    # TODO: use deepspeed's python api instead of subprocessing
    def _arg_if_present(var, var_name):
        """Need to wrap any arguments whose default value in train() is `None`"""
        if var:
            return f"--{var_name} {var}"
        return " "

    call(
        "deepspeed "
        + num_gpus_flag
        + " --module training.trainer --deepspeed "
        + deepspeed_config
        + f" --train_data={str(train_data)}"
        + f" --weights={input_model}"
        + f" --num_train_epochs={num_train_epochs}"
        + f" --max_steps={max_steps}"
        + _arg_if_present(eval_data, "eval_data")
        + f" --learning_rate {learning_rate}"
        + f" --train_batch_size {train_batch_size}"
        + f" --gradient_accumulation_steps {gradient_accumulation_steps}"
        + f" --logging_steps {logging_steps}"
        + f" --warmup_ratio {warmup_ratio}"
        + f" --gradient_checkpointing {gradient_checkpointing}"
        + " --local_output_dir "
        + output_dir,
        shell=True,
    )

    if os.path.exists(MODEL_OUT):
        os.remove(MODEL_OUT)

    model = T5ForConditionalGeneration.from_pretrained(output_dir, torch_dtype=torch.float16)
    serializer = TensorSerializer(MODEL_OUT)
    serializer.write_module(model)
    serializer.close()

    return TrainingOutput(weights=Path(MODEL_OUT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on a text dataset"
    )
    parser.add_argument(
        "--train_data", type=Path, required=True, help="Path to the json dataset"
    )
    parser.add_argument(
        "--eval_data",
        type=Path,
        required=False,
        help="Path to the json dataset",
        default=None,
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
        default=1,
        help="Number of training steps to run, overrides num_train_epochs, useful for testing",
    )
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        type=bool, 
        default=True, 
        help="Path to deepspeed config file."
    )
    some_args = parser.parse_args()
    train(**vars(some_args))
