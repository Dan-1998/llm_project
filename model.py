from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./finetuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Function to run inference
def run_inference(input_text):
    # Prepare the input data
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Generate predictions
    output_sequences = model.generate(**inputs)
    
    # Decode the output sequences to readable text
    decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return decoded_output

# Example usage
input_text = "Given the tweet 'I'm feeling sick'; return a list of possible hashtags separated by spaces"
predicted_output = run_inference(input_text)
print("Predicted Output:", predicted_output)