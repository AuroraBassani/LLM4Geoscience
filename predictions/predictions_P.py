from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch
from transformers import (
    AutoTokenizer,
    pipeline
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset


model_id = './tiny-llama'
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, load_in_8bit=False,
                                             device_map="auto",
                                             trust_remote_code=True)

model_path = "./fine-tuning/tinyllama_ft_P"

peft_model = PeftModel.from_pretrained(model, model_path, from_transformers=True, device_map="auto")

model = peft_model.merge_and_unload()


# get tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained(model_id)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"

def formatted_prompt(path):
    data = load_dataset('csv', data_files = path, split="train")
    data_df = data.to_pandas()
    data_df["text"] = data_df.apply(lambda x: "<|user|>\n" + x["Input"] + "</s>\n<|assistant|>", axis = 1)
    data = Dataset.from_pandas(data_df)
    return data

dataset_path_test = "./dataset/test_p.csv"
test_data = formatted_prompt(dataset_path_test)

# initialize pipeline
text_gen = pipeline(task="text-generation", model=model, tokenizer=llama_tokenizer, max_new_tokens = 100)

predictions = []

for out, row in zip(tqdm(text_gen(KeyDataset(test_data, 'text'), max_new_tokens=40)), test_data):

    prediction = out[0]['generated_text'].split('<|assistant|>\n')[1]
    predictions.append(prediction)


test_df = test_data.to_pandas()
test_df['prediction'] = predictions
test_df.to_csv('./predictions/predictions_P.csv')
