# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:40:36 2019

@author: PUMATHUR
"""

#Properties

EXTRACTOR_LOG_FILE_PATH = "C:\\Users\\PUMATHUR\\Desktop\\Cod\\ExtractorLogger.log"
MULTIPLY_FACTOR = 0.5


import PyPDF2
#This is for URL extraction using beautiful soup
import bs4 as bs
import urllib.request
#This is the Logger for the entire class
import logging
#Property file where paths are stored
#import settings
#This is regular expression package
import re
#This is NLTK package for all the text processing
import nltk
import heapq
import numpy as np
#Start of Corpus Class

"""class representing a collecting of text or a body of 
    writing on a particular subject. 
"""
class Corpus:
    positionWeights = []
     
    def __init__(self):
        self.sentences = []
        self.title = ""
        self.showSentences = []
        
    """"def __init__(self,title, sentences):
         self.sentences = sentences
         self.title = title
         self.showSentences = []"""
     
    def getSentences(self):
        return self.sentences
    
    def getTitle(self):
        return self.title
    
    def setSentences(self, sentences):
        self.__extractSentences(sentences)
        self.calcPositionWeights()
    
    def setTitle(self, title):
        self.title = title
    
    def __extractSentences(self, fullText):
        fullText = re.sub(r'\[[0-9]*\]',' ', fullText)
        fullText = re.sub(r'\s+',' ',fullText)
        clean_text = fullText.lower()
        clean_text = re.sub(r'\W', ' ', clean_text)
        clean_text = re.sub(r'\d', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text)
        self.sentences = nltk.sent_tokenize(fullText)
        #self.showSentences = nltk.tokenize(clean_text)
    
    def calcPositionWeights(self):
        counter = 1;
        for sentence in self.sentences:
            Corpus.positionWeights.append(counter * 0.1 * MULTIPLY_FACTOR)
            counter += 1
    
    def summarizer(self):
        dataset = self.sentences.copy()
        stop_words = nltk.corpus.stopwords.words('english')
        word2count = {}
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in stop_words:
                    if word not in word2count.keys():
                        word2count[word] = 1
                    else:
                        word2count[word] += 1
                        
        freq_words = heapq.nlargest(100, word2count, key=word2count.get)
        
        # IDF Matrix
        word_idfs = {}
        
        for word in freq_words:
            doc_count = 0
            for data in dataset:
                if  word in nltk.word_tokenize(data):
                    doc_count +=1
            word_idfs[word] =  np.log((len(dataset)/doc_count)+1)
            if word in self.title:
                # Increase the score by 0.5 for a word which is contained in the title
                word_idfs[word] = word_idfs[word] + MULTIPLY_FACTOR
            
        # TF Matrix
            
        tf_matrix = {}
        for word in freq_words:
            doc_tf = []
            for data in dataset:
                frequency = 0
                for w in nltk.word_tokenize(data):
                    if w == word:
                        frequency += 1
                tf_word = frequency/len(nltk.word_tokenize(data))
                doc_tf.append(tf_word)
            tf_matrix[word] = doc_tf
           
        # TF_IDF Calculation
            
        tfidf_matrix = []
        for word in tf_matrix.keys():
            tfidf = []
            for value in tf_matrix[word]:
                score = value * word_idfs[word]
                tfidf.append(score)
            tfidf_matrix.append(tfidf)
        
        X = np.asarray(tfidf_matrix)
        
        X = np.transpose(X)
      
        sent2score = {}
        
        for i in range (len(X)):
            sum = 0
            for j in range (len(X[i])):
                sum = sum + X[i][j]
            sent2score[dataset[i]] = sum + Corpus.positionWeights[i]
         
        best_sentences = heapq.nlargest(5,sent2score,key=sent2score.get)
        return best_sentences
        
#End of Corpus class

#Extractor class for PDf file and URL extraction
class Extractor:
    
    """This method extracts the PDf file and returns a dictionary with sentences
    INPUT: a list of pdf file paths
    OUTPUT: a dictionary of sentences
    """
    def extractPDF(self, path):
        corpus = Corpus()
        text = ""
        logging.info("----- START OF PDF EXTRACTOR PROCESSING -----")
        logging.basicConfig(filename=EXTRACTOR_LOG_FILE_PATH,level=logging.DEBUG)
        pdfFile = PyPDF2.PdfFileReader(path)
        if pdfFile.isEncrypted:
            pdfFile.decrypt()
        logging.info("File Info: " , pdfFile.documentInfo.items())
        for index, page in enumerate(pdfFile.pages):
            logging.info("----- PAGE " + str(index+1) + " -----")
            text += page.extractText() + " ";
        corpus.setSentences(text)
        corpus.setTitle(pdfFile.getDocumentInfo().title)
        logging.info("------ END OF PDF EXTRACTOR PROCESSING -----")
        return corpus
    
    def extractPDFs(self, paths):
        logging.info("----- START OF BULK PROCESSING -----")
        CorpusList = []
        for path in paths:
            CorpusList.append(self.extractPDF(path))
        logging.info("----- END OF BULK PROCESSING -----")
        return CorpusList    
    
    def extractURL(self, url):
        corpus = Corpus()
        logging.info("----- START EXTRACTOR URL PROCESSING -----")
        source = urllib.request.urlopen(url).read()
        soup = bs.BeautifulSoup(source, 'lxml')
        text = ""
        title = soup.find('h1').text
        for paragraph in soup.find_all('p'):
            text += paragraph.text
        corpus.setTitle(title)
        corpus.setSentences(text)
        logging.info("------ END OF EXTRACTOR URL PROCESSING -----")
        return corpus
    
#End of Extractor class
        
# Path where the file to be extracted can be picked up   
path = "C:\\Users\\PUMATHUR\\Desktop\\Cod\\resume-samples.pdf"
PDFExtractor = Extractor()
corpus = PDFExtractor.extractPDF(path)
SummarizedText = corpus.summarizer()
for sentence in SummarizedText:
    print(sentence)
print("\n")
#URLExtrcator = Extractor.Extractor()

# According to TF-IDF model, if a sentence's length is short and it contains infrequent words, the weightage of the sentence is high, and consequnetly , the chances of it appearing in the summary are high