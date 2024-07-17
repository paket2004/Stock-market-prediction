import gensim.downloader as api
import os

project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(project_root_dir, 'models', 'word2vec-google-news-300.model')

def download_word2vec_model():
    print("Loading word2vec-google-news-300 model...")
    wv = api.load('word2vec-google-news-300')
    wv.save(model_path)
    print("Model saved as word2vec-google-news-300.model")

if __name__ == "__main__":
    download_word2vec_model()
