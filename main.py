import pyspark
import json
import re
import emoji

hashtags = []
tweets = []

hashtag_pattern = re.compile(r'#\w+')
with open('out.json') as json_file:
    extracted_data = []
    for line in json_file:
        tweet = json.loads(line).get('full_text', '')
        hashtags.append(hashtag_pattern.findall(tweet))
        tweets.append(tweet)

print(hashtags[0:10])
print(tweets[0:10])

print(len(hashtags))
print(len(tweets))

tweets = [tweet for i, tweet in enumerate(tweets) if 4 > len(hashtags[i]) > 0]
hashtags = list(filter(lambda x: 4 > len(x) > 0, hashtags))

print(hashtags[0:10])
print(tweets[0:10])
print(len(hashtags))
print(len(tweets))

counter = {}


for i, hashtag in enumerate(hashtags):
    if len(hashtag) not in counter.keys():
        counter[len(hashtag)] = 1
    else:
        counter[(len(hashtag))] += 1

print(counter)
print({key: counter[key] for key in sorted(counter)})


def remove_bad_tweet_data(data):
    return re.sub(re.compile(r'#\w+|@\S+:? |RT | ?, ?|http\S+| ?\| ?| ?\u2026 '
                             r'?'), '',
                  emoji.replace_emoji(data.replace('\n', '').strip(' ,'),
                                      '')).strip(' :|')


tweets = [remove_bad_tweet_data(tweet) for tweet in tweets]


def remove_bad_hashtag_data(data):
    return re.sub(re.compile(r'[\[\],\']'), '', data)


with open('training.csv', 'w', encoding='utf-8') as training_csv:
    assert len(hashtags) == len(tweets)
    # writer = csv.writer(training_csv)
    training_csv.write('Input,Output\n')
    for i in range(len(hashtags)):
        training_csv.write(f'Given the tweet "{tweets[i]}"; retur'
                           f'n a list of possible hashtags,'
                           f'{remove_bad_hashtag_data(str(hashtags[i])).strip(",")}')
        training_csv.write('\n')