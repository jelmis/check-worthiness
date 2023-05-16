from emoji import demojize
from nltk.tokenize import TweetTokenizer
import numpy as np
import re


tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


if __name__ == "__main__":
    print(
        normalizeTweet(
            "SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
        )
    )


def furtherNormalizeTweet(tweet, clip_tokenizer):

    tweet_no_emoji = re.sub("(^|\s):\S.*?\S:", "", tweet)
    tokens_no_emoji = clip_tokenizer(tweet_no_emoji, padding=True, return_tensors="pt")["input_ids"][0]
    seq_length = (tokens_no_emoji == 49407).nonzero(as_tuple=True)[0][0] + 1  # length of sequence inclunding <EOS>

    if seq_length > 77:
        # Remove multiple occurunces of @USER and HTTPURL
        words = np.array(tweet_no_emoji.split())
        if "@USER" in words:
            words = words[~(words == "@USER")].astype(str).tolist()
            tweet_no_emoji = " ".join(words + ["@USER"])
        words = np.array(tweet_no_emoji.split())
        if "HTTPURL" in words:
            words = words[~(words == "HTTPURL")].tolist()
            tweet_no_emoji = " ".join(words + ["HTTPURL"])

    return tweet_no_emoji
