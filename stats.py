import json
import re
from geopy.geocoders import Nominatim
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import itertools
from wordcloud import WordCloud
from language_dict import language_dict

def _count_words_per_tweet(tweets: list):
    return [len(tweet.split()) for tweet in tweets]

def _total_words(tweets: list):
    return sum(_count_words_per_tweet(tweets))

def _average_words_per_tweet(tweets: list):
    return _total_words(tweets) / len(tweets)

def _count_hashtags_per_tweet(hashtags: list):
    return [len(hashtag) for hashtag in hashtags]

def _total_hashtags(hashtags: list):
    return sum(_count_hashtags_per_tweet(hashtags))

def _count_average_tweets_per_language(laguages: list) -> dict:
    return {language: laguages.count(language) for language in set(laguages)}

def _number_of_users(users: list):
    return len(set([user['id'] for user in users]))

def _count_user_location(locations: list):
    return {location: locations.count(location) for location in set(locations)}

def _remove_non_english(text, pattern=r'[^a-zA-Z0-9\s]'):
    pattern = re.compile(pattern)
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def raw_dataset_statistics(json_file):
    hashtags = []
    tweets = []
    languages = []
    users = []
    locations = []

    geolocator = Nominatim(user_agent="big_data_project")

    hashtag_pattern = re.compile(r'#\w+')
    with open(json_file) as json_file:
        for line in json_file:
            tweet_json = json.loads(line)
            tweet = tweet_json.get('full_text', '')
            language = tweet_json.get('lang', '')
            user = tweet_json.get('user', '')
            # filetered_location = _remove_non_english(tweet_json['user']['location'])
            # location = geolocator.geocode(filetered_location, timeout=2)

            hashtags.append(hashtag_pattern.findall(tweet))
            tweets.append(tweet)
            languages.append(language)
            users.append(user)
            # locations.append(location)

    words_per_tweet = _count_words_per_tweet(tweets)
    total_words = _total_words(tweets)
    average_words = _average_words_per_tweet(tweets)
    hashtags = [hashtag for tweet_hashtags in hashtags for hashtag in tweet_hashtags]
    hashtags_per_tweet = _count_hashtags_per_tweet(hashtags)
    total_hashtags = _total_hashtags(hashtags)
    average_tweets_per_language = _count_average_tweets_per_language(languages)
    number_of_users = _number_of_users(users)
    # user_location = _count_user_location(users)

    raw_dataset_statistics = {
        'words_per_tweet': _count_words_per_tweet(tweets),
        'total_words': _total_words(tweets),
        'average_words': _average_words_per_tweet(tweets),
        'hashtags': hashtags,
        'hashtags_per_tweet': _count_hashtags_per_tweet(hashtags),
        'total_hashtags': _total_hashtags(hashtags),
        'average_tweets_per_language': _count_average_tweets_per_language(languages),
        'number_of_users': _number_of_users(users)
        # 'user_location': _count_user_location(users)
    }
    return raw_dataset_statistics

def language_pie_chart(average_tweets_per_language: dict):
    df = pd.DataFrame(average_tweets_per_language.items())
    df.columns = ["language", "values"]
    df['language'] = df['language'].map(language_dict)
    df = df.sort_values(by='values', ascending=False)
    fig = px.pie(df, values='values', names='language')
    fig.update_traces(textposition='inside')
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    fig.show()

def hashtags_wordcloud(hashtags: list):
    all_hashtags = [hashtag for hashtag in hashtags]
    str_hashtags = " ".join(all_hashtags)
    wc = WordCloud(background_color='white',
                   max_words=100,
                   colormap='binary',
                   width=800,
                   height=500).generate_from_text(str_hashtags)
    plt.axis("off")
    plt.imshow(wc)
    plt.show()

def tweet_stats_table(raw_dataset_statistics: dict):
    selected_stats = {key: raw_dataset_statistics[key] for key in ["total_words", "average_words", "total_hashtags", "number_of_users"]}
    df = pd.DataFrame(selected_stats.items())
    df.columns = ["Statistic", "Value"]
    df["Value"] = df["Value"].round(2)
    return df

if __name__ == '__main__':
    raw_dataset_statistics = raw_dataset_statistics('out.json')
    language_pie_chart(raw_dataset_statistics["average_tweets_per_language"])
    hashtags_wordcloud(raw_dataset_statistics["hashtags"])
    print(tweet_stats_table(raw_dataset_statistics))


