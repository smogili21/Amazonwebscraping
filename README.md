# Amazonwebscraping

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup 

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer,cosine_distance
import numpy as np
from sklearn import mixture

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc,precision_recall_curve

from sklearn.svm import SVC

headers = ({'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
            AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/90.0.4430.212 Safari/537.36',
            'Accept-Language': 'en-US, en;q=0.5'})

def getReviews(page_url):        
           

        last_page = 6;
        
        rows= []
        for page in range(1,last_page):
            page_url_fin = page_url % page
            soup = BeautifulSoup(requests.get(page_url_fin, headers=headers).content, 'html.parser') 
            review_all =soup.find("div",class_="a-section a-spacing-none review-views celwidget")

            for idx, div in enumerate(review_all):
                
                name=None
                rating=None
                date=None
                review=None
                     
        
                p_name=div.select("span.a-profile-name")
                if p_name!=[]:
                    name=p_name[0].get_text() 
        
                p_review= div.select("span.review-text-content")
                if p_review!=[]:
                    review= p_review[0].get_text().replace('\n', '').strip() 
                
                p_rating= div.select("span.a-icon-alt")
                if p_rating!=[]:
                    rating= p_rating[0].get_text().replace('.0 out of 5 stars', '').strip()
                    
                p_date= div.select("span.review-date")
                if p_date!=[]:
                    date= p_date[0].get_text().replace('Reviewed in the United States on ', '').strip()
                    
                
        
                rows.append((name,date,review,rating))

        reviews = pd.DataFrame(rows, columns=('Name','Date','Review','Rating',))

        return reviews    


# In this section we are scrapping reviews from Amazon.com for Iphone 12, Iphone 12 pro and Iphone 12 pro max 
# We will be creating a DataFrame with Name, Date , Review and Rating

page_url_1 = "https://www.amazon.com/Apple-iPhone-Graphite-Carrier-Subscription/product-reviews/B08L5NHRWN/ref=cm_cr_getr_d_paging_btm_next_3?ie=UTF8&reviewerType=all_reviews&pageNumber=%d"
reviews_1=getReviews(page_url_1)

page_url_2 ="https://www.amazon.com/Apple-iPhone-Locked-Carrier-Subscription/product-reviews/B08L5P7DYY/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=%d"
reviews_2=getReviews(page_url_2)

page_url_3 ="https://www.amazon.com/Apple-iPhone-Pro-128GB-Graphite/product-reviews/B08PL89SJS/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=%d"
reviews_3=getReviews(page_url_3)


reviews = reviews_1.append(reviews_2, ignore_index=True)

reviews = reviews.append(reviews_3, ignore_index=True)

reviews = reviews[reviews.Name.notnull()]


reviews['Rating'] = reviews['Rating'].astype(int)

reviews.reset_index(drop=True, inplace=True)

reviews

# We are diving the dataset into two lists Positive and Negative for further Sentimental Analysis


positive_list = reviews[reviews.Rating > 2]
negative_list = reviews[reviews.Rating < 3]

#positive_list.shape
#negative_list.shape

def word_list(positive_negative_list):
    review_list= []
    for x in range(len(positive_negative_list.index)):
        review_list.append(positive_negative_list.values[x][2])
    
    text = ". ".join(review_list).lower()
    
    return text


positive_word_list = word_list(positive_list)

negative_word_list = word_list(negative_list)


import nltk, re, json, string
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import spacy
from nltk.probability import FreqDist


def most_common_words(text, K,speech):
    stop_words = stopwords.words('english')
   
    tokens=[token.strip() \
            for token in nltk.word_tokenize(text.lower()) \
            if token.strip() not in stop_words and\
               token.strip() not in string.punctuation]
    
    
    tagged_tokens= nltk.pos_tag(tokens)
    
    #print(tagged_tokens)
    
    words = [token[0] for token in tagged_tokens if token[1] in [speech]]

    fdist=nltk.FreqDist(words) 
    
    result = fdist.most_common(K)
    
    return result

# We are analysing the Text for most frequent words and specifications details for positive reviews

print(most_common_words(positive_word_list,20,'NNS'))

print(most_common_words(positive_word_list,20,'NN'))

print(most_common_words(positive_word_list,20,'JJ'))

print(most_common_words(positive_word_list,20,'RB'))


# We are analysing the Text for most frequent words and specifications details for negative reviews

print(most_common_words(negative_word_list,20,'NNS'))

print(most_common_words(negative_word_list,20,'NN'))

print(most_common_words(negative_word_list,20,'JJ'))

print(most_common_words(positive_word_list,20,'RB'))


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image


wordcloud_negative = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(negative_word_list)

wordcloud_positive = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(positive_word_list)

plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis("off")
plt.show()


#img = wordcloud.to_image()
#img.show()

from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

sentiment_positive = sid.polarity_scores(positive_word_list)

sentiment_negative = sid.polarity_scores(negative_word_list)

print(sentiment_positive)
print(sentiment_negative)

from gensim.models import word2vec
import logging
import pandas as pd

stop_words = stopwords.words('english')

sentences=[ [token.strip(string.punctuation).strip() \
             for token in nltk.word_tokenize(doc.lower()) \
                 if token not in string.punctuation and \
                 token.strip() not in stop_words and \
                 len(token.strip(string.punctuation).strip())>=2]\
             for doc in reviews["Review"]]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

wv_model = word2vec.Word2Vec(sentences, \
            min_count=5, vector_size=200, \
            window=5, workers=4 )

print("Top 5 words similar to word 'screen'")
wv_model.wv.most_similar('screen', topn=5)

reviews['Label'] = reviews['Rating'].apply(lambda x: 0 if x <= 2 else 1)

train = reviews[:90]
test = reviews[90:]

train_text = train["Review"]
test_label = test["Label"]
test_text = test["Review"]

train.head()



tfidf_vect = TfidfVectorizer(min_df=1, stop_words='english')
    
dtm= tfidf_vect.fit_transform(train_text)
    
num_clusters=2

clusterer = KMeansClusterer(num_clusters, \
                                cosine_distance, \
                                repeats=20)

clusters = clusterer.cluster(dtm.toarray(), \
                             assign_clusters=True)
test_dtm = tfidf_vect.transform(test_text)
 
predicted = [clusterer.classify(v) for v in test_dtm.toarray()]



confusion_df = pd.DataFrame(list(zip(test_label.values, predicted)),\
                                columns = ["label", "cluster"])
    
confusion_df_fin = pd.crosstab( index=confusion_df.cluster, columns=confusion_df.label)



cluster_dict={0:1,1:0}


predicted_target=[cluster_dict[i] \
                      for i in predicted]

print(metrics.classification_report\
     (test_label, predicted))
    

def create_model_final(train_docs, train_y, test_docs, test_y,
              model_type, stop_words, min_df,max_df,ngram_range, print_result, algorithm_para):
    
    vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=min_df, max_df = max_df, ngram_range = ngram_range)
    train_docs = vectorizer.fit_transform(train_docs)
    
    if model_type == 'svm':
        clf = svm.LinearSVC().fit(train_docs, train_y)
        test_docs = vectorizer.transform(test_docs)
        predicted=clf.predict(test_docs)
        print(classification_report(test_y, predicted))
        
        model = SVC(kernel="linear", probability=True)
        model.fit(train_docs, train_y)

        decision_scores = model.decision_function(test_docs)
        fpr, tpr, thresholds = roc_curve(test_y, decision_scores,pos_label=1)
        auc_score = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(test_y, decision_scores,pos_label=1)
        prc_score = auc(recall, precision) 
        
        print("AUC: {:.2%}".format(auc(fpr, tpr)), " PRC: {:.2%}".format(auc(recall, precision)))
        
        plt.figure();
        plt.plot(fpr, tpr, color='darkorange', lw=2);
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
        plt.xlim([0.0, 1.0]);
        plt.ylim([0.0, 1.05]);
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('AUC of SVM');
        plt.show();

        plt.figure();
        plt.plot(recall, precision, color='darkorange', lw=2);
        plt.xlim([0.0, 1.0]);
        plt.ylim([0.0, 1.05]);
        plt.xlabel('Recall');
        plt.ylabel('Precision');
        plt.title('PRC of SVM');
        plt.show();
    
         
        
    return auc_score, prc_score

auc_score, prc_socre = create_model_final(train["Review"], train["Label"], test["Review"], test["Label"], \
          model_type='svm', stop_words = None, min_df =2, max_df =0.75 ,ngram_range=(1, 2), print_result=True, algorithm_para=0.3)

