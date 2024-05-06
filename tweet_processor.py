import json
import re
import csv

# Function to clean tweet text
def clean_tweet(tweet):
    # Remove "RT" if it starts with it
    tweet = re.sub(r'^RT\s', '', tweet)
    # Remove @ mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove quotation marks
    tweet = tweet.replace('"', '').replace("'", "")
    # Remove unicode characters and emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # Remove commas
    tweet = tweet.replace(',', '')
    # Remove unnecessary whitespaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    # Remove all https links
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    # Normalize colons followed by several spaces
    tweet = re.sub(r':\s+', '', tweet)
    # Normalize pipe characters followed by several spaces
    tweet = re.sub(r'\|\s+', '', tweet)
    return tweet

# Function to extract and keep hashtags
def extract_hashtags(tweet):
    hashtags = re.findall(r'(#\w+)', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    return tweet.strip(), ' '.join(hashtags)

# Function to process each tweet
def process_tweet(tweet_data):
    full_text = tweet_data["full_text"]
    hashtags = [hashtag['text'] for hashtag in tweet_data["entities"]["hashtags"]]

    if 1 <= len(hashtags) <= 5:
        cleaned_text = clean_tweet(full_text)
        cleaned_text, extracted_hashtags = extract_hashtags(cleaned_text)
        if len(cleaned_text) >= 4 and len(extracted_hashtags) >= 1:
            input_text = f"Given the tweet '{cleaned_text}'; return a list of possible hashtags separated by spaces"
            return [input_text, extracted_hashtags]
    return None

# Read JSON data and process each entry
results = []
with open("tweets.json", "r", encoding="utf-8") as file:
    for line in file:
        try:
            tweet_data = json.loads(line)
            result = process_tweet(tweet_data)
            if result:
                results.append(result)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)

# Write results to CSV
with open("tweets.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input", "Output"])  # Write headers
    writer.writerows(results)  # Write all processed tweets

print("CSV file 'tweets.csv' has been created and populated based on specified criteria.")
