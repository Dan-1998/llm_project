import pandas as pd
from datasets import Dataset
# <<<<<<< HEAD
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler, \
    AutoModelForMaskedLM, T5Tokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt

model_name = "t5-small"#"Twitter/twhin-bert-large"#"t5-large"


class Trainer:
    def __init__(self):
        self.df = None
        self.dataset = None
        self.tokenizer = None
        self.split_datasets = None
        self.trainer = None
        self.model = None

    def load(self):
        print("LOAD")
        # Load the data
        self.df = pd.read_csv('tweets.csv')
        self.dataset = Dataset.from_pandas(self.df)
# =======
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_scheduler
# import matplotlib.pyplot as plt
#
# # Function to prepocess tweets for GPU implementation
# def preprocess_function(examples):
#     inputs = [tweet for tweet in examples['Input']]
#     targets = [hashtags.replace("#", "") for hashtags in examples['Output']]
#     model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
#
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
# >>>>>>> 73fbd128f51e5a5ebd434450fd179bb5fa665849

    def preprocess(self):
        print("PREPROCESS")
        # Preprocess the data
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)#AutoTokenizer.from_pretrained(model_name)

# <<<<<<< HEAD
    def tokenize(self):
        print("TOKENIZE")
        # Tokenize the dataset
        tokenized_datasets = self.dataset.map(self.preprocess_function,
                                              batched=True)

        # Split the dataset into training and evaluation sets
        self.split_datasets = tokenized_datasets.train_test_split(test_size=0.1)

    def train(self):
        print("LOAD MODEL")
        # Load a pre-trained model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name) if model_name.startswith('t5') else T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        #AutoModelForMaskedLM.from_pretrained(model_name)#

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
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.split_datasets['train'],
            eval_dataset=self.split_datasets['test'],
            tokenizer=self.tokenizer
        )

        print("TRAIN")
        # Train the model
        self.trainer.train()

    def post_process(self):
        print("PLOT LOSS")
        # Plot the training loss
        loss_values = self.trainer.state.log_history
        plt.figure(figsize=(10, 5))
        plt.plot([x['loss'] for x in loss_values if 'loss' in x],
                 label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print("SAVE")
        # Save the model
        self.model.save_pretrained(f"./{model_name}")
        self.tokenizer.save_pretrained(f"./{model_name}")

    def run(self):
        self.load()
        self.preprocess()
        self.tokenize()
        self.train()
        self.post_process()

    def preprocess_function(self, examples):
        inputs = [tweet for tweet in examples['Input']]
        targets = [hashtags.replace("#", "") for hashtags in examples['Output']]  # Remove '#' from hashtags
        model_inputs = self.tokenizer(inputs, max_length=128, truncation=True,
                                  padding="max_length")

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=128, truncation=True,
                                padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run()
# =======
# # Load  data and run tokenizer
# print("LOAD")
# df = pd.read_csv('tweets.csv')
# dataset = Dataset.from_pandas(df)
#
# print("TOKENIZE")
# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
# split_datasets = tokenized_datasets.train_test_split(test_size=0.2)
#
# # Training parameters
# print("LOAD MODEL")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=3,
#     predict_with_generate=True,
#     logging_dir='./logs',
#     logging_strategy='steps',
#     logging_steps=10,
# )
#
# # Trainer initilization and run command
# print("LOAD TRAINER")
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=split_datasets['train'],
#     eval_dataset=split_datasets['test'],
#     tokenizer=tokenizer
# )
#
# print("TRAIN")
# trainer.train()
#
# # Plot training loss visualization
# print("PLOT LOSS")
# loss_values = trainer.state.log_history
# plt.figure(figsize=(10, 5))
# plt.plot([x['loss'] for x in loss_values if 'loss' in x], label='Training Loss')
# plt.title('Training Loss Over Time')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # Save model and end script
# print("SAVE")
# model.save_pretrained("./finetuned_model")
# tokenizer.save_pretrained("./finetuned_model")
# >>>>>>> 73fbd128f51e5a5ebd434450fd179bb5fa665849
