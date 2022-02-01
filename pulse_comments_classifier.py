#PREPROCESSING STAGE

import pandas as pd
import numpy as np
import re
import nltk
# nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

df = pd.read_excel("sample.xlsx")
df = df.dropna()
comments = df['Comment']
categories = df['Category'].reset_index(drop=True)

#General Cleaning
cleaned = comments.apply(lambda x: re.sub("'s",'',x)) #remove all possessive word (?)
puncnum = '!"#$%&()*+, -./:;<=>?@[\]^_`{|}~' + '0123456789' + "'"
t = str.maketrans(dict.fromkeys(puncnum," "))
cleaned = cleaned.apply(lambda x: x.translate(t)) #remove all punctuation and numbers
cleaned = cleaned.apply(lambda x: re.sub('\r|\s{2,}',' ',x)) #replace all return carriages and multiple whitespace with single whitespace
cleaned = cleaned.apply(lambda x: x.strip()) #strip whitespaces
del t

#Load glossary
glossary = pd.read_excel("glossary.xlsx")
glossary = glossary[glossary.Change == 1].drop('Change',axis=1).reset_index(drop=True)

#List of all non-ambiguous acronyms: Non ambiguous abbreviations, Full, Others 1, Others 2
non_ambi = pd.concat([glossary['Full'],glossary['Others_1'].dropna(),
                      glossary['Others_2'].dropna(),
                      glossary[pd.isnull(glossary['Ambiguous'])]['Abbreviation']],axis=0).dropna().tolist()

non_ambi = sorted(non_ambi,key=lambda x: len(x),reverse=True)

#List of all ambiguous acronyms
ambi = glossary[glossary['Ambiguous']==1]['Abbreviation'].tolist()
ambi = sorted(ambi,key=lambda x: len(x),reverse=True)

def acr_ex(comments,glossary=glossary):
    """
    Substitutes acronyms and alternate spellings in each comment with the desired output word
    """
    expanded = []
    for i in range(len(comments)):
        com = comments[i]
        ambi_exp = []
        exp = []
        
        def find_ind(acronym,regex):
            if bool(regex.search(com)):
                ind = np.where(glossary.isin([acronym]))[0][0]
                return ind
            else:
                return None
                
        for acr in non_ambi:
            reg = re.compile(r'\b%ss?\b'%acr,re.IGNORECASE)
            try:
                ind = find_ind(acr,reg)
                if ind is not None:
                    exp.append((reg.search(com).group(),ind))   
            except:
                pass
        
        for acr in ambi:
            reg = re.compile(r'\b%ss?\b'%acr)
            try:
                ind = find_ind(acr,reg)
                if ind is not None:
                    ambi_exp.append((re.search(r'\b%ss?\b'%acr,com).group(),ind))     
            except:
                pass
        
        acrsort = lambda x: len(x[0])
        ambi_exp.sort(key=acrsort,reverse=True)
        exp.sort(key=acrsort,reverse=True)
        
        for lst in [ambi_exp,exp]:
            flags = 0 if lst==ambi_exp else re.IGNORECASE
            for acr,ind in lst:
                com = re.sub(acr,glossary.loc[ind,'Final'],com,flags=flags)
        expanded.append(com.lower())
        
    return expanded

#Expand acronyms
X = acr_ex(np.array(cleaned))

#Define stop words
stop_words = list(set(stopwords.words('english')))
stop_words = [i for i in stop_words if i not in ['not','no']]

#Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

for n in range(len(X)):
    com = nltk.regexp_tokenize(X[n],pattern='\s+',gaps=True)
    com = [w for w in com if w not in stop_words]
    com = [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in com]
    X[n] = ' '.join(com)
del lemmatizer, com, w

#Convert categories to numerical labels
y = categories.factorize()[0]
category_id = pd.Series(y,name='category_id')
category_id_df = pd.concat([categories,category_id],axis=1).drop_duplicates()
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['Category', 'category_id']].values)
del category_id, category_id_df, category_to_id

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=76)
comments_train,comments_test,_,_ = train_test_split(comments, y, test_size=0.2,random_state=76)

#TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
tfidf = TfidfVectorizer(use_idf=True, min_df=5,max_df=0.4,ngram_range=(1, 3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
tfidf_keys = tfidf.get_feature_names_out()
tfidf_dict = dict(zip(tfidf_keys,tfidf.idf_))
del tfidf_keys


#TRAINING STAGE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score

#Train data using default settings of different classifiers
np.random.seed(1234)
X_cv = X_train_tfidf
y_cv = y_train

dual = False if X_cv.shape[0]>X_cv.shape[1] else True
logsolver = 'liblinear' if dual else 'lbfgs'

classifiers = [LogisticRegression(dual=dual,class_weight='balanced',solver='liblinear'),
               LinearSVC(dual=dual),
               SVC(class_weight='balanced'),
               KNeighborsClassifier(),
               RandomForestClassifier(),
               GradientBoostingClassifier(),
               MultinomialNB()]

clf_names = ['Logistic Regression','LinearSVC','SVC','KNN',
             'Random Forest','Gradient Boost','Naive Bayes']

clf_default_scores = {}

#Output accuracies in a list
for clf,name in zip(classifiers,clf_names):
    print(name)
    cvscore = cross_validate(clf,X_cv,y_cv,cv=10,verbose=2,scoring='accuracy')
    clf_default_scores[name] = np.mean(cvscore['test_score'])

del clf_names, logsolver

#Hyperparameter Tuning
import warnings
from sklearn.exceptions import ConvergenceWarning,FitFailedWarning
warnings.simplefilter("ignore", category=FitFailedWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

def hp_tuner(models,params,names,x=X_cv,y=y_cv):
    for i in range(len(models)):
        gridsearch=GridSearchCV(estimator=models[i], param_grid=params[i],
                                n_jobs=-1, cv=10, verbose=2,
                                scoring='accuracy').fit(x,y)
        globals()['gs_%s' % names[i]]=pd.DataFrame(gridsearch.cv_results_)

def hp_best_result(results):
    best_results = []
    for result in results:
        best = result.loc[result['rank_test_score']==1,'params':'std_test_score'].reset_index(drop=True)
        best_results.append(best.T.squeeze())
        
    best_df = pd.DataFrame(best_results)
    return best_df

#Run 1
models = [LogisticRegression(),LinearSVC(),SVC()]
model_log=['logistic1','linearsvc1','svc1']

logistic_params = {'C':[0.1,1,10],'solver':['newton-cg','lbfgs','saga','sag','liblinear'],
                   'penalty':['l1','l2','elasticnet'],'max_iter':[50000],'dual':[dual]}
linearsvc_params = {'C':[0.1,1,10],'dual':[dual]}
svc_params = {'C':[0.1,1,10],'kernel':['linear','poly','rbf','sigmoid']}
param_grid = [logistic_params,linearsvc_params,svc_params]

hp_tuner(models,param_grid,model_log)
gs1_best = hp_best_result([gs_logistic,gs_linearsvc,gs_svc])

#Run 2
models = [LogisticRegression(),LinearSVC(),SVC()]
model_log=['logistic2','linearsvc2','svc2']

logistic_params = {'C':[0.5,1,10,20,50],'solver':['newton-cg','lbfgs','saga','sag','liblinear'],
                   'penalty':['l2'],'max_iter':[50000],'dual':[dual]}
linearsvc_params = {'C':[0.05,0.1,0.5],'dual':[dual]}
svc_params = {'C':[0.1,1,10],'kernel':['linear','poly','rbf','sigmoid']}
param_grid = [logistic_params,linearsvc_params,svc_params]

hp_tuner(models,param_grid,model_log)
gs2_best = hp_best_result([gs_logistic2,gs_linearsvc2,gs_svc2])

#Run 3
models = [LinearSVC()]
model_log=['linearsvc3']

linearsvc_params = {'C':np.arange(0.07,0.20,0.01),'dual':[dual]}
param_grid = [linearsvc_params]

hp_tuner(models,param_grid,model_log)
gs3_best = hp_best_result([gs_linearsvc3])

#Predict test results
from sklearn.metrics import classification_report, confusion_matrix
model = LinearSVC(C=0.13)
model.fit(X_cv,y_cv)

y_pred = model.predict(X_test_tfidf)
pred_results = classification_report(y_test,y_pred)
    
pred_cm = pd.DataFrame(confusion_matrix(y_test,y_pred),
                       columns=id_to_category,index=id_to_category)
