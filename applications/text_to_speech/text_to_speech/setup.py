import nltk
import os

def setup():
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("cmudict")
    os.system("python -m unidic download")
