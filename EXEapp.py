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
from rich.console import Console
from rich.progress import Progress
from rich.text import Text



#nltk.download("stopwords")
console = Console()

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

custom_responses = {
    "hi": "Hello! How can I assist you today?",
    "hey": "Hey there! What can I do for you?",
    "hello": "Hi! Need any help?",
    "yo": "Yo! What's going on? Need a hand?",
    "sup": "Sup! How can I help?",
    "what's up": "What's up? Need anything?",
    "good morning": "Good morning! How can I help you start your day?",
    "good afternoon": "Good afternoon! What can I do for you today?",
    "good evening": "Good evening! How can I assist you this evening?",
    "how do you do": "How do you do? It's a pleasure to assist you.",
    "pleased to meet you": "Pleasure's mine! How can I help?",
    "nice to meet you": "Nice to meet you too! Let's get started.",
    "can you help me": "Absolutely! What do you need help with?",
    "what can you do": "I can do many things! searches, analyzes, and summarizes content from the web",
    "how do I": "Let's figure it out together! How can I help?",
    "tell me about": "I'd be happy to! Tell me what you're interested in.",
    "do you know": "Let's find out! Ask away.",
    "how are you": "I'm doing well, thanks for asking! How can I help you?",
    "thank you": "You're welcome! Is there anything else I can do for you?",
    "see you later": "See you later! Don't hesitate to contact me.",
    "thanks for your help": "You're very welcome! Let me know if you need anything else.",
    "help": "You can type any query or URL to get a summarized response.",
    "about": "I am Avc(Anveshanam Vishleshanam Ch) a chatbot Develpoed by Shiva in year 2024, that searches, analyzes, and summarizes content from the web."
}


# Main execution
if __name__ == "__main__":
    # Welcome message
    console.print("Hi, welcome! I am Avc, an AI chatbot that searches, analyzes, and summarizes. Please ask your query or direct URL that you want me to summarize.", style="bold cyan")
    console.print(f"[bold yellow]> Bot:[/bold yellow] Hi, how can i help you ? [bold red](Hint : enter bye to exit)[/bold red]", style="bold cyan")

    while True:
        query = console.input("[bold yellow]> You: [/bold yellow]").lower()
        if query == 'bye':
            console.print("[bold yellow]> Bot:[/bold yellow] Goodbye! Have a great day.", style="bold green")
            break
        elif query in custom_responses:
            console.print(f"[bold yellow]> Bot:[/bold yellow] {custom_responses[query]}", style="bold green")
        else:
            results = search_duckduckgo(query)
            if not results:
                console.print("[bold yellow]> Bot:[/bold yellow] No results found.", style="bold red")
            else:
                selected_results = results[:1]  # Choose the first result        
                for url in selected_results:
                    summary = summarize_url(url, summary_sentences=5, method='lsa')  # You can use 'lex_rank' or 'frequency' as alternative methods
                    console.print(f"[bold yellow]> Bot:[/bold yellow] {summary}", style="bold green")
                    console.print(f"[bold blue]URL:[/bold blue] {url}")
                    console.print("\n" + "="*100 + "\n")
