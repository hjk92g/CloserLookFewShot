#It is modified from a code in https://gist.github.com/johnnyasd12/9442bcbe00f470c8546c528480f701b6

import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re
import shutil



cwd = os.getcwd() 
print(cwd)

try:
    os.mkdir('preprocessed')
except:
    pass

try:
    imbg_folders=os.listdir('images_background')
    imev_folders=os.listdir('images_evaluation')
    #print(imbg_folders)
    for f_nm in imbg_folders:
        shutil.move('images_background/'+f_nm,'preprocessed')
    for f_nm in imev_folders:
        shutil.move('images_evaluation/'+f_nm,'preprocessed')
except:
    pass


data_path = join(cwd, 'preprocessed') 
savedir = './'
dataset_list = ['base', 'val', 'novel']

cl = -1
folderlist = [] # to store label??

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} } # label1:[fname1,fname2,...], label2:[fname...], ...
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

for dataset in dataset_list:
    with open(datasetmap[dataset] + ".csv", "r") as lines: # read train.csv, val.csv, test.csv
        for i, line in enumerate(lines):
            if i == 0:
                continue
            fid, _ , label = re.split(',|\.', line) # fid here: filename before .jpg
            label = label.replace('\n','')
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = [] # new label
                fnames = listdir( join(data_path, label) ) # preprocessed files names.jpg in this class
                fname_number = [ int(re.split('_|\.', fname)[0]) for fname in fnames] # preprocessed files names before.jpg
                sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0] # this class files names.jpg
            name = fid[-8:] + '.jpg'
            fname = join( data_path,label, name ) # file path, BUGFIX: sorted_fnames[fid]
            filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist() 

for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END) 
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
