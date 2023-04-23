import string
import nltk
import pandas as pd
from collections import Counter
from nltk.stem.porter import PorterStemmer
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt



def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


L = []


# myDF = pd.DataFrame(columns=['tokenized'])
id = 0
with open('little_prince.txt', 'r') as fin, open('tokens.txt','w') as fout:  # Open file for read
    for line in fin:           # Read line-by-line
        # Process the line
        if line.isspace(): 
            continue

        line = line.lower()
        result = line.translate(str.maketrans('','', string.punctuation))
        result = result.strip()
        result = result.replace('\n', '')

        result = ''.join([i for i in result if not i.isdigit()])

        result = remove_stopwords(result) # remove stopwords


        tokens = nltk.word_tokenize(result)
        #tokens = ''.join([i for i in tokens if i.isalpha()])
        tokens = ' '.join(tokens)
        L.append(tokens)
        
        # id = id + 1
        # print(tokens)s
        #print(' '.join(tokens), end='\n', file=fout) 
# File closed automatically upon exit of with-statement


df = pd.DataFrame(L, columns=['tokens'])
#print(df.head(10))

stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

df["text_stemmed"] = df["tokens"].apply(lambda text: stem_words(text))
# print(df.head(10))

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

df["html_removed"] = df["tokens"].apply(lambda text: remove_html(text))
#print(df.head(10))

huge_str = ''
stopwords = set(STOPWORDS)

filtered = []
for val in df.html_removed:
    filtered.append(val)

huge_str = ' '.join(filtered)

wordcloud = WordCloud(width = 800, height = 800,
            background_color ='white',
            stopwords = stopwords,
            min_font_size = 10).generate(huge_str)

plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()