This repository is an implementation of a fine-tunable [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

The model can be fine tuned using `cog train`, and can serve predictions using `cog predict`. 

All that `cog train` requires is an input dataset consisting of a JSON list where each example has a 'prompt' and 'completion' field. The model will be fine-tuned to produce 'completion' given 'prompt'. Here's an example command to train the model from the root directory:

```
cog train -i train_data="https://storage.googleapis.com/dan-scratch-public/fine-tuning/70k_samples_prompt.jsonl" -i gradient_accumulation_steps=8 -i learning_rate=2e-5 -i num_train_epochs=3 -i logging_steps=2 -i train_batch_size=4
```

Of the params above for training, the only required param is the `train_data`, but you can pass other parameters to modify training the model as you see fit. See the 'examples' folder for an example dataset.

This project also has the ability to build and push a cog container for any of the FLAN family of models. Just run `cog run python select_model.py --model_name ["flan-t5-small" "flan-t5-base" "flan-t5-large" "flan-t5-xl" "flan-t5-xxl" "flan-ul2"]`, and then you can run all other `cog` commands with the appropriate model. 
