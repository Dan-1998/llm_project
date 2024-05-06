import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
import matplotlib.pyplot as plt

print("LOAD")
# Load the data
df = pd.read_csv('tweets.csv')
dataset = Dataset.from_pandas(df)

print("PREPROCESS")
# Preprocess the data
tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_function(examples):
    inputs = [tweet for tweet in examples['Input']]
    targets = [hashtags.replace("#", "") for hashtags in examples['Output']]  # Remove '#' from hashtags
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("TOKENIZE")
# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and evaluation sets
split_datasets = tokenized_datasets.train_test_split(test_size=0.1)

print("LOAD MODEL")
# Load a pre-trained model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir='./logs',  # To store logs
    logging_strategy='steps',
    logging_steps=10,  # Log every 10 steps
)

print("LOAD TRAINER")
# Initialize the Trainer with a custom callback for plotting
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_datasets['train'],
    eval_dataset=split_datasets['test'],
    tokenizer=tokenizer
)

print("TRAIN")
# Train the model
trainer.train()

print("PLOT LOSS")
# Plot the training loss
loss_values = trainer.state.log_history
plt.figure(figsize=(10, 5))
plt.plot([x['loss'] for x in loss_values if 'loss' in x], label='Training Loss')
plt.title('Training Loss Over Time')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("SAVE")
# Save the model
model.save_pretrained("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")