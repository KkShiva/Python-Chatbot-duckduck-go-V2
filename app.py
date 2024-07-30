import requests
from bs4 import BeautifulSoup
import random
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from collections import defaultdict
from string import punctuation
from heapq import nlargest
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from flask import Flask, request, jsonify, render_template

#nltk.download("stopwords")
app = Flask(__name__)

class FrequencySummarizer:
    def __init__(self, min_cut=0.1, max_cut=0.9):
        self._min_cut = min_cut
        self._max_cut = max_cut
        self._stopwords = set(stopwords.words('english') + list(punctuation))

    def _compute_frequencies(self, word_sent):
        freq = defaultdict(int)
        for s in word_sent:
            for word in s:
                if word not in self._stopwords:
                    freq[word] += 1
        m = float(max(freq.values()))
        for w in list(freq):
            freq[w] = freq[w] / m
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                del freq[w]
        return freq

    def summarize(self, text, n):
        sents = sent_tokenize(text)
        if n > len(sents):
            n = len(sents)
        word_sent = [word_tokenize(s.lower()) for s in sents]
        self._freq = self._compute_frequencies(word_sent)
        ranking = defaultdict(int)
        for i, sent in enumerate(word_sent):
            for w in sent:
                if w in self._freq:
                    ranking[i] += self._freq[w]
        sents_idx = self._rank(ranking, n)
        return [sents[j] for j in sents_idx]

    def _rank(self, ranking, n):
        return nlargest(n, ranking, key=ranking.get)

def search_duckduckgo(query):
    url = 'https://html.duckduckgo.com/html/'
    params = {'q': query}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    response = requests.post(url, data=params, headers=headers)
    if response.status_code != 200:
        print(f"Error: Unable to fetch search results (status code: {response.status_code})")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for link in soup.find_all('a', class_='result__a'):
        results.append(link.get('href'))
    return results

def get_text_from_url(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text

def summarize_text(text, summary_sentences=5, method='lsa'):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    if method == 'lsa':
        summarizer = LsaSummarizer()
    elif method == 'lex_rank':
        summarizer = LexRankSummarizer()
    else:
        raise ValueError("Invalid summarization method specified.")
    summary = summarizer(parser.document, summary_sentences)
    return " ".join(str(sentence) for sentence in summary)

def summarize_url(url, summary_sentences=5, method='lsa'):
    url_text = get_text_from_url(url).replace(u"Â", u"").replace(u"â", u"")
    if method == 'frequency':
        fs = FrequencySummarizer()
        summary = fs.summarize(url_text.replace("\n", " "), summary_sentences)
    else:
        summary = summarize_text(url_text.replace("\n", " "), summary_sentences, method)
    return summary



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    results = search_duckduckgo(query)
    if not results:
        return jsonify({'error': 'No results found'}), 404
    
    selected_result = results[0]
    summary = summarize_url(selected_result, summary_sentences=5, method='lsa')
    
    return jsonify({'Abstract': summary, 'AbstractSource': selected_result})

if __name__ == '__main__':
    app.run(debug=True)

