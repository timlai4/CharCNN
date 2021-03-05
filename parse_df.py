# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 22:09:15 2021

@author: Tim
"""

import pandas as pd

df = pd.read_csv("data.csv").astype('float32')
df.rename(columns={'0':'label'}, inplace=True)

letter_dic = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',
                    8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',
                    15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',
                    22:'W',23:'X',24:'Y',25:'Z'}
df_alphabets = df.copy()
df['label'] = df['label'].map(letter_dic)
ACTG = df[df['label'].isin(['A','C','T','G'])].sample(frac=1).reset_index(drop=True)

split = int(0.15*len(ACTG))
cv = ACTG.head(split)
train = ACTG.tail(len(ACTG) - split)

assert len(cv) + len(train) == len(ACTG)

cv.to_csv("CV.csv", index = False)
train.to_csv("train.csv", index = False)