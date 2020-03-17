"""
1. Generates data according to first letter or mail
2. data = from 100 pairs of points (a, a+noise), a[-1] = 9999
3. Requires:
1) Is it lineary dependent?
2) Is it lineary dependent if we remove outliers?
3) Kendall?
4) Spearman?
"""
import numpy as np
import scipy.stats as st
from collections import Counter
from scipy.stats import jarque_bera
from scipy.stats import shapiro
from scipy.stats import binom_test
from zlib import crc32
from statsmodels.stats.weightstats import ttest_ind
import itertools
from arch.bootstrap import IIDBootstrap
from statsmodels.stats.multitest import multipletests

def generate(seed):    
    rs = np.random.RandomState(seed)
    answers = {}
    alpha = rs.uniform()*10
    data = rs.randn(100,2)
    data[:,1] = data[:,0]
    data[:,1] +=  rs.randn(100)*alpha
    data[-1,0] = 99999
    
    answers['pearson'] = (st.pearsonr(data[:,0], data[:,1])[1]<0.05)*1
    answers['pearson2'] = (st.pearsonr(data[:-1,0], data[:-1,1])[1]<0.05)*1
    answers['kendall'] = st.kendalltau(data[:,0], data[:,1])[0]
    
    return data, answers



from tqdm import tqdm as tqdm

if __name__=='__main__':
    with open('./mails_all.csv') as inp:
        add_emails = inp.readlines()
    add_emails = list(set([s.strip().lower().split('@')[0] for s in add_emails]))
    with open('answers.csv', 'w') as out:
        first = True
        for word in tqdm(list('abcdefghijklmnopqrstuvwxyz')+add_emails):
            h = crc32(word.encode('utf-8'))
            seed = h%(2**32-1)

            data, answers = generate(seed)
            np.savetxt('./datas/' + word+".csv", data, delimiter=",")
            if first:
                out.write('index,' + ','.join(sorted(list(answers.keys())))+'\n')
                first = False
            out.write(word+',' + ','.join([str(answers[w]) for w in sorted(list(answers.keys()))])+'\n')
            
