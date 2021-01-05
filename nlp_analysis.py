import nltk
import itertools

# silently grab all nltk dependencies
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from requests.exceptions import HTTPError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import wordnet, stopwords

lemmatizer = nltk.WordNetLemmatizer(); next(wordnet.words())
stemmer = nltk.stem.porter.PorterStemmer()

def leaves(tree):
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()

def normalise(word):
    word = word.lower()
    #word = lemmatizer.lemmatize(word)
    return word

def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf ]
        yield term

def get_noun_phrases(text):
        sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*|CD>}  # Nouns and Adjectives, terminated with Nouns
                
            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>}
        """
        chunker = nltk.RegexpParser(grammar)

        toks = nltk.regexp_tokenize(text, sentence_re)
        postoks = nltk.tag.pos_tag(toks)
        tree = chunker.parse(postoks)
        
        terms = get_terms(tree)
        phrases = [" ".join(term) for term in terms]
        return phrases

def get_wordnet_pos(tag):
    if tag.startswith('ADJ'):
        return wordnet.ADJ
    elif tag.startswith('VERB'):
        return wordnet.VERB
    elif tag.startswith('NOUN'):
        return wordnet.NOUN
    elif tag.startswith('ADV'):
        return wordnet.ADV
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else: return None

def tag_pos(phrase):
    return [(x[0], get_wordnet_pos(x[1])) for x in nltk.tag.pos_tag(nltk.tokenize.word_tokenize(phrase)) if x[1] not in ('.','DET')]

def get_distance(w1, w2, pos = None, def_cutoff:int = 2):
    # https://stackoverflow.com/questions/30829382/check-the-similarity-between-two-words-with-nltk-with-python
    if not isinstance(w1, list): w1 = [w1]
    if not isinstance(w2, list): w2 = [w2]
    sims = list(); kwargs = {'pos': pos} if pos else {}
    for word1, word2 in itertools.product(w1, w2):
        syns1 = list(filter(lambda sn: def_cutoff >= int('0'+''.join(x for x in repr(sn) if x.isdigit())), wordnet.synsets(word1, **kwargs)))
        syns2 = list(filter(lambda sn: def_cutoff >= int('0'+''.join(x for x in repr(sn) if x.isdigit())), wordnet.synsets(word2, **kwargs)))
        if len(syns1) * len(syns2) == 0: continue
        for sense1, sense2 in itertools.product(syns1, syns2):
            d = wordnet.wup_similarity(sense1, sense2)
            if d == None: d = 0
            sims.append((1-d, sense1, sense2))
    if len(sims) <= 0: return (0, None, None)
    return min(sims, key=lambda x: x[0])

def get_min_lingual_distance(w, lst, pos=None):
    if not isinstance(lst, list): lst = [lst]
    if len(lst) <= 0: return w
    best = (lst[0], get_distance(w, lst[0],pos=pos)[0])
    for i in range(1,len(lst)):
        dst = get_distance(w, lst[i], pos=pos)
        if dst[0] < best[1]: best = (lst[i], dst[0])
        elif dst[0] == best[1]: # if the distance is the same (most often 0 and 0)...
            # if edit distance is less, that's the better option
            if nltk.edit_distance(best[0], w)/max(len(best[0]),1) > nltk.edit_distance(lst[i], w)/max(len(lst[i]),1): best = (lst[i], dst[0]) 
    return best

def process_string(s):
    s = stemmer.stem(str(s).lower().strip().strip(':').strip())
    return s

def get_min_distance(w, lst, preprocess=process_string, pos=None):
    p_w = preprocess(w)
    best = (None, 1)
    for lw in lst:
        dist = get_distance(w, lw, pos=pos)
        if not isinstance(dist[1], nltk.corpus.reader.wordnet.Synset) or not isinstance(dist[1], nltk.corpus.reader.wordnet.Synset): 
            if len(w) > 0: dist = [nltk.edit_distance(p_w, preprocess(lw))/len(lw)]
            else: continue
        if dist[0] < best[1]: best = (lw, dist[0])
    if best[0] == None: return (lst[0], 1)
    return best

def get_min_edit_distance(string, iterable, length_dependant:bool=True, preprocess=process_string):
    string = preprocess(string)
    iterable = list(filter(lambda x: x != None, iterable))
    distances = sorted({s : nltk.edit_distance(string, preprocess(s)) / (max(len(preprocess(s)),0.01) if length_dependant else 1) for s in iterable}.items(), key=lambda i: i[1])
    if len(distances) > 0: return tuple(distances[0])
    return (string, 0)

if __name__ == "__main__":
    print(get_min_distance('snake',['dab','es','spain']))
    print(tag_pos('I stop my heartbeat when I see myself in what these women do'))