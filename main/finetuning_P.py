from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from peft import get_peft_model


# Tokenizer path
tokenizer_path = './tiny-llama'

# base model path
base_model_name = "./tiny-llama" 
# new model path
new_model_name = "./fine-tuning/tinyllama_ft_P" 
# Tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

# base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto"
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1




def prepare_data(data_id):
    data = load_dataset('csv', data_files = data_id, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df[["Input", "Output"]].apply(lambda x: "<|user|>\n" + x["Input"] + "</s>\n<|assistant|>\n" + x["Output"] + "\n</s>", axis=1)
    data = Dataset.from_pandas(data_df)
    return data

# training dataset
dataset_path_train = "./DataModel/dataset/train_p.csv"
training_data = prepare_data(dataset_path_train)

# validation dataset
dataset_path_val = "./DataModel/dataset/val_p.csv"
val_data = prepare_data(dataset_path_val)




# Training Parameterss
train_params = TrainingArguments(
    output_dir="./fine-tuning/results_P",
    overwrite_output_dir = True, #overwrite the content of the output directory
    num_train_epochs=30,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1, 
    optim="paged_adamw_32bit",
    save_strategy = 'epoch', # save after each epoch
    logging_strategy = 'epoch', # show loss after each epoch 
    learning_rate=4e-4,
    weight_decay=0.001,
    fp16=True, 
    bf16=False, 
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    # validation parameters
    evaluation_strategy="epoch",  # evaluate on validation after each epoch
    per_device_eval_batch_size  = 4,
    # save best model 
    load_best_model_at_end = True,
    metric_for_best_model = 'eval_loss',
    greater_is_better = False
)


# LoRA Configuration
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.4,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, peft_parameters)

print('summary trainable parameters: ', model.print_trainable_parameters())



# fine tuning configuration
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=training_data,
    peft_config=peft_parameters,
    dataset_text_field="text",
    tokenizer=llama_tokenizer,
    args=train_params,
    eval_dataset = val_data,
    max_seq_length = 2048 
)

# Training
fine_tuning.train()

# save fine tuned model 
fine_tuning.model.save_pretrained(new_model_name)
