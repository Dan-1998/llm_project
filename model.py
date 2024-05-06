from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./finetuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def run_inference(input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    print("Tokenized inputs:", inputs)  # Diagnostic print

    # Generate output sequences
    output_sequences = model.generate(**inputs, max_length=50, min_length=1, no_repeat_ngram_size=2)
    print("Raw output sequences:", output_sequences)  # Diagnostic print

    # Decode the output sequences to readable text
    decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return decoded_output

# Example usage
input_text = "Your input text here"
predicted_output = run_inference(input_text)
print("Predicted Output:", predicted_output)