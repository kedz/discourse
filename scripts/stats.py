from os.path import join, exists
from os import listdir
import re
import pandas as pd

def stats(dataname):
    data = []
    d = '/home/chris/projects/discourse/results'
    for f in listdir(d):
        eval = join(d, f, '{}-eval.txt'.format(dataname))
        if exists(eval):
            import os
            if os.path.getctime(eval) < 1393349473.85:
                continue
            
            x = []
            x.append(f.replace('role_match', 'rm')
                      .replace('is_first','if')
                      .replace('is_last', 'il')
                      .replace('first_word_ne', 'fw_ne')
                      .replace('use_det', 'dt')
                      .replace('use_subs', 'sub')
                      .replace('person', 'per')
                      .replace('num_caps', 'caps')
                      )
            fl = open(eval, 'r')
            lines = fl.readlines()       
            fl.close()
            
            ndocs = int(lines[0].split(': ')[-1])
            x.append(ndocs)
            
            acc = float(lines[2].split(': ')[-1])
            x.append(acc)
            k = float(lines[3].split('Tau ')[-1])
            x.append(k)
            pval = float(lines[4].split(': ')[-1])
            x.append(pval)
            pw_corr = float(lines[5].split(': ')[-1])
            x.append(pw_corr)
            
            data.append(x)
    
    data.sort(key=lambda x: x[3])
            
    return pd.DataFrame(data, columns=['feat', 'ndocs', 'acc', 'kendalls tau', 'pval', 'pw_acc'])

print 'NTSB-TEST'
print stats('ntsb')
print 
print 'NTSB-TRAIN'
print stats('ntsb-train')
print 
print 'APWS-TEST'
print stats('apws')
print
print 'APWS-TRAIN'
print stats('apws-train')
