from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the trained model and tokenizer
model_path = "./finetuned_model_facebook_bart-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# def run_inference(input_text):
#     # Tokenize input
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
#     print("Tokenized inputs:", inputs)  # Diagnostic print
#
#     # Generate output sequences
#     output_sequences = model.generate(**inputs, max_length=50, min_length=1, no_repeat_ngram_size=2)
#     print("Raw output sequences:", output_sequences)  # Diagnostic print
#
#     # Decode the output sequences to readable text
#     decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
#     return decoded_output

def run_inference(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    output_sequences = model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        temperature=1.0,  # Adjust for more randomness
        top_k=0,  # Turn off top-k filtering
        top_p=1.0,  # Turn off nucleus sampling
        early_stopping=False  # Allow more freedom in generation
    )

    decoded_output = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return decoded_output

# Example usage
def eval_tweet(tweet):
    prompt = tweet if tweet.startswith('Given the tweet') else f'Given the ' \
                                                               f"tweet '" \
                                                               f"{tweet}'; " \
                                                               f"return a " \
                                                               f"list of " \
                                                               f"possible " \
                                                               f"hashtags " \
                                                               f"seperated by spaces"
    print('Input:', prompt)
    predicted_output = run_inference(prompt)
    print('Predicted Output:', predicted_output)

    return predicted_output

eval_tweet("Given the tweet 'I'm feeling sick with something bad'; return a " \
         "list of possible hashtags separated by spaces")

eval_tweet("Given the tweet 'My family is dying'; return a list of possible " \
             "hashtags " \
             "separated by spaces")

eval_tweet("Given the tweet 'When comparing competitors Moderna has a " \
             "market cap of " \
"$64B while Beam has a market cap of $3B (21x less). Being at the forefront of new tech creates lasting demand as Moderna was maker of one of COVID vaccines while Beam is still working on final products'; return a list of possible " \
             "hashtags " \
             "separated by spaces")

eval_tweet('The government screwed us over big time')

eval_tweet('Joe Biden is a vegetable at this point')

eval_tweet("l'incroyable implication du Mossad en collaboration avec Epstein, Bill Gates et ses milliards, l'OMS, l'institut Rockfeller, l'Union européenne, Bain & Company et la raison pour laquelle les médias se sont violemment attaqués au professeur Didier Raoult.")