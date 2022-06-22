import PIL
import requests
from urllib import request
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import cv2
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import os


class ContentAnalyzer:
    def __init__(self, url_list: list, queries: list):
        self.url_list = url_list
        self.queries = queries
        self.STOPWORDS = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.dir = '/Users/polarized/PycharmProjects/InfromationRetrieval/images/'
        self.model = VGG16()

    @staticmethod
    def read_html(url: str) -> str:
        '''
        Read url to html and save string
        :param url:
        :return:
        '''
        opener = request.URLopener({})
        resource = opener.open(url)
        return resource.read().decode(resource.headers.get_content_charset())

    @staticmethod
    def get_body_content(html: str) -> tuple:
        '''
        Collect text and images from html into lists
        :param html:
        :return:
        '''
        soup = BeautifulSoup(html, features='html.parser').find("div", {"id": "bodyContent"})
        content = []
        images = []
        for tag in soup.find_all('td', {'class': 'mbox-image'}):
            tag.replaceWith('')
        for tag in soup.find_all('span', {'class': 'navigation'}):
            tag.replaceWith('')
        for tag in soup.find_all():
            if tag.name == 'p':
                for sub in tag.find_all('span', {'class': 'mwe-math-element'}):
                    sub.replaceWith('')
                content.append(tag.text.strip('\n'))
            elif tag.name == 'img':
                if any(s in tag['src'][2:] for s in ('png', 'jpg')):
                    images.append('http://' + tag['src'][2:])
        return ' '.join(content), images

    def content_filter(self, content: str) -> list:
        '''
        Filtering text content from stopwords, brackets, nums and etc.
        :param content:
        :return:
        '''
        tokens = []
        for word in word_tokenize(content):
            word = word.lower()
            if word not in self.STOPWORDS and word.isalpha():
                tokens.append(word)
        return tokens

    def token_lemmatizer(self, tokens: list) -> list:
        '''
        Lemmatization every token
        :param tokens:
        :return:
        '''
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def content_analyzer(self, url: str) -> list:
        '''
        Handling url (read url, get content, image sum, filter, lemmatization)
        :param url:
        :return:
        '''
        html = self.read_html(url)
        content, images = self.get_body_content(html)
        self.download_images(images)
        images_sum = self.image_summarization()
        content += images_sum
        tokens = self.content_filter(content)
        lemma_tokens = self.token_lemmatizer(tokens)
        self.remove_images()
        return lemma_tokens

    def fill_corpus(self) -> list:
        corpus = []
        for i in range(0, len(self.url_list)):
            content = self.content_analyzer(self.url_list[i])
            corpus.append(content)
        return corpus

    def query_filter(self, queries: list) -> list:
        filtered_queries = []
        for q in range(0, len(queries)):
            q = self.content_filter(queries[q])
            filtered_queries.append(q)
        return filtered_queries

    def tokens_to_text(self, tokens: list) -> list:
        tmp = []
        for i in range(0, len(tokens)):
            tmp.append(' '.join(tokens[i]))
        return tmp

    def download_images(self, images: list) -> None:
        '''
        Saving images to directory self.dir
        Make sure you changed it before run
        :param images:
        :return:
        '''
        for i in range(0, len(images)):
            try:
                response = requests.get(images[i]).content
                np_arr = np.frombuffer(response, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                cv2.imwrite(self.dir + 'img_' + str(i) + '.jpg', img)
            except (cv2.error, requests.exceptions.InvalidURL) as exc:
                pass

    def image_summarization(self) -> str:
        '''
        Get image from directory self.dir and summarize into str
        :return:
        '''
        images_sum = []
        directory = os.fsencode(self.dir)
        for file in os.listdir(directory):
            try:
                filename = os.fsdecode(file)
                filename = self.dir + filename
                image = load_img(filename, target_size=(224, 224))
                image = img_to_array(image)
                image = image.reshape(
                    (1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                yhat = self.model.predict(image)
                label = decode_predictions(yhat)
                label = label[0][0]
                images_sum.append(label[1])
            except PIL.UnidentifiedImageError:
                pass
        return ' '.join(images_sum)

    def remove_images(self) -> None:
        '''
        Remove all files from self.dir
        :return:
        '''
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
