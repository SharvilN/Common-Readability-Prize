import warnings

from pandas.core.indexing import check_bool_indexer
from transformers.utils.dummy_pt_objects import DataCollatorForLanguageModeling, LineByLineTextDataset, Trainer
warnings.filterwarnings("ignore")

import pandas as pd

from transformers import AutoModelForMaskedLM, AutoTokenizer,TrainingArguments, Trainer

# create target text corpus
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

data = pd.concat([train, test], axis=0)

data.loc[:, "excerpt"] = data.excerpt.apply(lambda x: x.replace("\n", ""))
excerpts = "\n".join(data.excerpt.values.tolist()) 

with open("data/processed/pretrain_data.txt", "w") as f:
    f.write(excerpts)


# create model
checkpoint = "roberta-large"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


#create dataset
dtrain = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/processed/pretrain_data.txt",
    block_size=256
)

dvalid = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/processed/pretrain_data.txt",
    block_size=256
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="models/checkpoints/clrp_roberta_large_v1", #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
    report_to = "none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dtrain,
    eval_dataset=dvalid
)

trainer.train()
trainer.save_model("models/clrp_roberta_large_v1")