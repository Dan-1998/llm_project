import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
import matplotlib.pyplot as plt

# Function to prepocess tweets for GPU implementation
def preprocess_function(examples):
    inputs = [tweet for tweet in examples['Input']]
    targets = [hashtags.replace("#", "") for hashtags in examples['Output']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load  data and run tokenizer
print("LOAD")
df = pd.read_csv('tweets.csv')
dataset = Dataset.from_pandas(df)

print("TOKENIZE")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenized_datasets = dataset.map(preprocess_function, batched=True)
split_datasets = tokenized_datasets.train_test_split(test_size=0.2)

# Training parameters
print("LOAD MODEL")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir='./logs',
    logging_strategy='steps',
    logging_steps=10,
)

# Trainer initilization and run command
print("LOAD TRAINER")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_datasets['train'],
    eval_dataset=split_datasets['test'],
    tokenizer=tokenizer
)

print("TRAIN")
trainer.train()

# Plot training loss visualization
print("PLOT LOSS")
loss_values = trainer.state.log_history
plt.figure(figsize=(10, 5))
plt.plot([x['loss'] for x in loss_values if 'loss' in x], label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save model and end script
print("SAVE")
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")