from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import csv
from afinn import Afinn
import sys

DEBUG = False

infilename = sis.argv[1]

VADERNEGTHRES = -0.45
VADERPOSTHRES = 0.35

AFINNNEGTHRES = 0
AFINNPOSTHRES = 0

def read_csv_file(filename, debug = False):
    '''

    :param filename:
    :param debug:
    :return:
    '''
    with open(filename) as f:
        reader = csv.reader(f, dialect="excel-tab")
        data = [row for row in reader]

    data = np.array(data)

    if debug:
        print('Data loaded. Shape: {}'.format(data.shape))

    return data


def do_VADER_SA(sentences, negthres, posthres):
    analyzer = SentimentIntensityAnalyzer()
    annotated = []

    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        compound = vs['compound']
        if compound < negthres:
            annotated.append('n')
        elif compound > posthres:
            annotated.append('p')
        else:
            annotated.append('d')

    return annotated



def do_afinn_SA(sentences, negthres, posthres):
    analyzer = Afinn()
    annotated = []

    for sentence in sentences:
        compound = analyzer.score(sentence)

        if compound < negthres:
            annotated.append('n')
        elif compound > posthres:
            annotated.append('p')
        else:
            annotated.append('d')

    return annotated

def make_confmat(listA, listB):
    '''

    :param listA:
    :param listB:
    :return:
    '''

    all_labels = set(list(listA) + list(listB))
    all_labels = ['p', 'd', 'n']
    labeldict = {label:i for i, label in enumerate(all_labels)}
    print(labeldict)
    nr_unique = len(set(list(listA) + list(listB)))

    #sortedlabels = sorted(list(labeldict.keys()))

    m = np.zeros((nr_unique, nr_unique))

    for true, pred in zip(listA, listB):
        m[labeldict[true]][labeldict[pred]] += 1

    return m


def get_confmatstring(m):
    confmatstr = \
        """
            predicted
            p   d   n
true    p   {}  {}  {}
        d   {}  {}  {}
        n   {}  {}  {}

        """.format(int(m[0, 0]), int(m[0, 1]), int(m[0, 2]),
                   int(m[1, 0]), int(m[1,1]), int(m[1,2]),
                   int(m[2,0]), int(m[2,1]), int(m[2,2]))
    return confmatstr


def print_confmat(m):
    confmatstr = get_confmatstring(m)
    print(confmatstr)

def get_pr_rec_string(m):
    precision_a = m[0, 0] / (m[0, 0] + m[1, 0] + m[2,0])
    recall_a = m[0, 0] / (m[0, 0] + m[0, 1] + m[0,2])
    precision_b = m[2, 2] / (m[2, 2] + m[0, 2] + m[1,2])
    recall_b = m[2, 2] / (m[2, 2] + m[2, 0] + m[2,1])


    pr_rec_string = \
        """
precision p:\t{}
recall p:\t{}
precision n:\t{}
recall n:\t{}
        """.format(precision_a, recall_a, precision_b, recall_b)

    return pr_rec_string

def evaluate_sentiment_analysis(m):
    precision_positive = m[0, 0] / (m[0, 0] + m[1, 0] + m[2,0])
    print('precision pos:\t{}'.format(precision_positive))
    recall_positive = m[0, 0] / (m[0, 0] + m[0, 1] + m[0,2])
    print('recall pos:\t{}'.format(recall_positive))

    precision_neg = m[2, 2] / (m[2, 2] + m[0,2]+ m[1,2])
    print('precision neg:\t{}'.format(precision_neg))

    recall_neg = m[2, 2] / (m[2, 2] + m[2, 0] + m[2,1])
    print('recall neg:\t{}'.format(recall_neg))


def print_error_analysis(Ypred, Ytrue, posts, m, logfilename = 'error_analysis.txt'):

    log = open(logfilename, 'wt')
    log.write('p: positive\n')
    log.write('n: negative\n')
    log.write('d: neutral\n')


    log.write('{}\n'.format(get_confmatstring(m)))

    log.write('{}\n'.format(get_pr_rec_string(m)))


    log.write('---pred = n and true != n  --------------------------------------------\n\n')
    i = 0
    for true, pred in zip(Ytrue, Ypred):
        if pred == 'n' and true == 'p':
            #print()
            #print(posts[i])
            log.write('\ntrue: {} - pred: {}\n{}\n'.format(true, pred, posts[i]))

        i+=1

    log.write('\n\n---pred = p and true != p  --------------------------------------------\n\n')
    i = 0
    for true, pred in zip(Ytrue, Ypred):
        if pred == 'p' and true == 'n':
            #print()
            #print(posts[i])
            log.write('\ntrue: {} - pred: {}\n{}\n'.format(true, pred, posts[i]))

        i+=1

    log.close()



golden = read_csv_file(infilename)
header_golden = golden[0]
topiclabels = golden[1:,1]
forumposts = golden[1:,5]

vader_SA_labs = do_VADER_SA(forumposts, VADERNEGTHRES, VADERPOSTHRES)

m = make_confmat(topiclabels, vader_SA_labs)
print(get_confmatstring(m))

evaluate_sentiment_analysis(m)

print_error_analysis(vader_SA_labs, topiclabels, forumposts, m, logfilename='error_analysis_VADER.txt')



Afinn_SA_labs = do_afinn_SA(forumposts, AFINNNEGTHRES, AFINNPOSTHRES)
m = make_confmat(topiclabels, Afinn_SA_labs)
print(get_confmatstring(m))

evaluate_sentiment_analysis(m)
print_error_analysis(Afinn_SA_labs, topiclabels, forumposts, m, logfilename='error_analysis_AFINN.txt')
