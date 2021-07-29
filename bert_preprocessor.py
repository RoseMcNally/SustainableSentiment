import re
import wordninja
import emoji


class BertPreprocessor:

    @staticmethod
    def process_hashtags(text: str, brackets=True):
        tags = re.findall(r'#\S+', text)
        tags = list(set(tags))
        tags = sorted(tags, key=len, reverse=True)
        for tag in tags:
            word = " ".join(wordninja.split(tag[1:]))
            if brackets:
                word = '<' + word + '>'
            text = re.sub(re.escape(tag), word, text)
        return text

    @staticmethod
    def process_emojis(text, angle_brackets=True):
        text = emoji.demojize(text)
        emo_text = re.findall(r':+[\S]+:', text)
        emo_text = list(set(emo_text))
        for emo in emo_text:
            word = emo[1:-1].replace("_", " ")
            if angle_brackets:
                word = "<" + word + ">"
            else:
                word = "(" + word + ")"
            text = re.sub(re.escape(emo), word, text)
        return text

    @staticmethod
    def bert_text_preprocess(text, sentiment_preprocess=False):
        text = re.sub(r'http\S+', 'url', text)
        text = re.sub(r'@', '', text)
        text = re.sub('&amp;', 'and', text)
        if sentiment_preprocess:
            text = BertPreprocessor.process_hashtags(text, False)
            text = BertPreprocessor.process_emojis(text, False)
        else:
            text = BertPreprocessor.process_hashtags(text)
            text = BertPreprocessor.process_emojis(text)
        return text

    @staticmethod
    def preprocess(data):
        data['Text'] = data['Text'].apply(lambda x: BertPreprocessor.bert_text_preprocess(x))
        return data
