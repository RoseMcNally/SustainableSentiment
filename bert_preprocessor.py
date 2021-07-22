import re
import wordninja
import emoji


class BertPreprocessor:

    @staticmethod
    def process_hashtags(text: str):
        tags = re.findall(r'#\S+', text)
        tags = list(set(tags))
        tags = sorted(tags, key=len, reverse=True)
        for tag in tags:
            word = '<' + " ".join(wordninja.split(tag[1:])) + '>'
            text = re.sub(re.escape(tag), word, text)
        return text

    @staticmethod
    def process_emojis(text):
        text = emoji.demojize(text)
        emo_text = re.findall(r':+[\S]+:', text)
        emo_text = list(set(emo_text))
        for emo in emo_text:
            word = "<" + emo[1:-1].replace("_", " ") + ">"
            text = re.sub(emo, word, text)
        return text

    @staticmethod
    def bert_text_preprocess(text):
        text = re.sub(r'http\S+', 'url', text)
        text = re.sub(r'@', '', text)
        text = re.sub('&amp;', 'and', text)
        text = BertPreprocessor.process_hashtags(text)
        text = BertPreprocessor.process_emojis(text)
        return text

    @staticmethod
    def preprocess(data):
        data['Text'] = data['Text'].apply(lambda x: BertPreprocessor.bert_text_preprocess(x))
        return data
