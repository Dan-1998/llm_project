import pyspark
import json
import re

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

tweets = [re.sub(hashtag_pattern, '', tweet.replace('\n', '')).strip() for
          tweet in
          tweets]

with open('training.csv', 'w', encoding='utf-8') as training_csv:
    assert len(hashtags) == len(tweets)
    # writer = csv.writer(training_csv)
    for i in range(len(hashtags)):
        training_csv.write(f'human: {tweets[i]} \\n bot: {hashtags[i]}')
        training_csv.write(',\n')

string = "RT @matt7gh: 3/\n\nricordiamo come i giornali #mainstream diffondevano menzogne sull'immunità naturale da #Covid per pompare i finti #vaccini…"
print(string)
print(string.replace('\n', ''))
# hashtags = []
# with open('hashtags_and_urls.txt', encoding='utf-8') as twitter_file:
#     hashtags = [line.strip('\n') for line in twitter_file.readlines() if
#                 line.startswith(
#         '#')]
#
# print(hashtags)