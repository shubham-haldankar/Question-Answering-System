import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary= dict()
    for file in os.listdir(directory):
        path= os.path.join(directory,str(file))
        with open(path,encoding='utf8') as f:
            dictionary[file]= f.read()
    return dictionary
    #raise NotImplementedError


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    lowercase= document.lower()
    tokenized= nltk.word_tokenize(lowercase)
    stopwords= nltk.corpus.stopwords.words("english")
    words= []
    for word in tokenized:
        if word not in string.punctuation and word not in stopwords:
            words.append(word)
    return words
    #raise NotImplementedError


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words= set()
    for file in documents.keys():
        for word in documents[file]:
            words.add(word)
    idfs= dict()
    n= len(documents)
    for word in words:
        idfs[word]= math.log(n/sum([word in documents[file] for file in documents.keys()]))
    return idfs
    #raise NotImplementedError


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs= dict()
    for name in files.keys():
        tfidf= 0
        for word in files[name]:
            if word in query:
                tfidf+= idfs[word]
        tfidfs[name]= tfidf
    return sorted(tfidfs.keys(),key=lambda x:tfidfs[x])[-n:][::-1]
    #raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    values= dict()
    for key in sentences.keys():
        words= sentences[key]
        idf= 0
        i= 0
        inner_dict= dict()
        for word in query:
            if word in words:
                idf+= idfs[word]
                i+= 1
        inner_dict['idf']= idf
        inner_dict['query_term_density']= i/len(words)
        values[key]= inner_dict
    return sorted(values, key= lambda x:(values[x]['idf'],values[x]['query_term_density']))[-n:][::-1]
    #raise NotImplementedError


if __name__ == "__main__":
    main()
