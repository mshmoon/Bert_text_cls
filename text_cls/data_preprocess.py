import re
import pandas as pd
from sklearn.utils import shuffle
import json

def clean(text):
    # text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)  # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)  # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)  # 去除网址
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()

def process_data(path):
    label = []
    content = []
    with open(path,"r",encoding="utf-8") as f:
        for single_data in f.readlines():
            single_data = json.loads(single_data)
            label.append(single_data["label"])
            text = clean(single_data["sentence"])
            content.append(text)

    return label,content


if __name__ == '__main__':
    train_path = 'dataset/train.json'
    process_data(train_path)
    print('Finish!')
