#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

## Kopal Garg, 1003063221

import numpy as np
import argparse
import json
import re
import sys
import string
import os
import csv


# Provided wordlists.
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

def return_num(x):
    if len(x) > 0:
        if x[0].isnumeric():
            return float(x)
    else:
        return np.NaN

# BristolGilhoolyLogie
bgl_data = {}
bgl_csv = open("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv")
bgl_reader = csv.DictReader(bgl_csv)
for row in bgl_reader:
    bgl_data[row["WORD"]] = [
            return_num(row["AoA (100-700)"]), 
            return_num(row["IMG"]), 
            return_num(row["FAM"])
                ]

# Warringer
warr_data = {}
warr_csv = open("/u/cs401/Wordlists/Ratings_Warriner_et_al.csv")
warr_reader = csv.DictReader(warr_csv)
for row in warr_reader:
    warr_data[row["Word"]] = [
            return_num(row["V.Mean.Sum"]),
            return_num(row["A.Mean.Sum"]), 
            return_num(row["D.Mean.Sum"])
            ]         
# labels/categories
a1_path = '/u/cs401/A1'
feats_path = os.path.join(a1_path, 'feats')

# LIWC
file_data = {
        'Left': np.array([0]),
        'Center': np.array([1]),
        'Right': np.array([2]),
        'Alt' : np.array([3])
        }
class_data = {
        'Left': 0,
        'Center': 1,
        'Right': 2,
        'Alt': 3
}
categories = ['Left', 'Center', 'Right', 'Alt']

for fname in file_data:
    tmp = {}
    ids = open(os.path.join(feats_path, fname + '_IDs.txt')).readlines() 
    liwc = np.load(os.path.join(feats_path, fname + '_feats.dat.npy'))
    for i in range(len(ids)):
        tmp[str(ids[i]).strip()] = liwc[i]
    file_data[fname] = tmp

def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''    
    # initalize feature vector of 173 length, which will be filled up till 29 and returned 
    
    feats = np.zeros(173)
    original_comment = comment
    # split the comment into body and POS
    body = re.compile("(\S+)/(?=\S+)").findall(comment) # before /
    POS = re.compile("(?<=\S)/(\S+)").findall(comment) # after /

    # TODO: Extract features that rely on capitalization.
    # feats[0] : Number of tokens in uppercase (??? 3 letters long)
    for i in body:
        if i.isupper() and len(i) >= 3:
            feats[0]+=1

    # TODO: Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN").
    for i in range(len(body)):
        body[i] = body[i].lower()
    # I will also convert body in comment to lower for convenient processing
    def replacement(match):
        return match.group().lower()
    comment = re.sub(r"[a-zA-Z]+/", replacement, comment)

    # TODO: Extract features that do not rely on capitalization.
    # feats[1] : Number of first-person pronouns
    # feats[2] : Number of second-person pronouns
    # feats[3] : Number of third-person pronouns
    for i in body:
        if i in FIRST_PERSON_PRONOUNS: feats[1]+=1
        if i in SECOND_PERSON_PRONOUNS: feats[2]+=1
        if i in THIRD_PERSON_PRONOUNS: feats[3]+=1
    # feats[4] : Number of coordinating conjunctions: 
    feats[4] = POS.count('CC')

    # feats[5] : Number of past-tense verbs: 
    feats[5] = POS.count('VBD')

    # feats[6] : Number of future-tense verbs: 
    # will, gonna, shall, 'll
    for i in body:
        if i.endswith("'ll") or i in {"will", "gonna", "shall"}:
            feats[6] +=1
    # going+to+vb
    pattern_ftb = re.compile(r"(?:go/VBG\s+to/[A-Z]{2,}\s+\w*/VB|going\s+to/[A-Z]{2,}\s+\w*/VB)")
    feats[6] = len(pattern_ftb.findall(comment))

    punct=string.punctuation
    for i in body:
        # feats[7] : Number of commas
        if i==",":
            feats[7] +=1
        # feats[8] : Number of multi-character punctuation tokens
        if len(i) > 1:
            if all([character in punct for character in i]):
                feats[8] +=1
       
    # feats[9] : Number of common nouns
    feats[9]=POS.count('NN')+POS.count('NNS')

    
    # feats[10] : Number of proper nouns
    feats[10]=POS.count('NNP')+POS.count('NNPS')


    # feats[11] : Number of adverbs
    feats[11] = POS.count('RB')+POS.count('RBR')+POS.count('RBS')


    # feats[12] : Number of wh- words
    pattern_wh = re.compile(r"(/WDT\b|/WP\$\W|/WRB\b|/WP\b)")
    feats[12] = POS.count('WDT')+POS.count('WP')+POS.count('WP$')+POS.count('WRB')

    # feats[13] : Number of slang acronyms
    pattern_slang = re.compile(r"(?:\s|^)("+r"|".join(SLANG)+")(?:/[A-Z]{0,4})")
    feats[13] = len(pattern_slang.findall(comment))

    sentences = comment.split("\n")[:-1]
    tokenCount=0
    
    if len(sentences) > 0:
        for sentence in sentences:
            tokenCount += len(sentence.split())
        # feats[14] : Average length of sentences, in tokens
        feats[14] = tokenCount / len(sentences)
    
    
    # feats[15] : Average length of tokens, excluding punctuation-only tokens, in characters
    num_tokens = 0
    length = 0
    for token in body:
        if not set(token).issubset(set(string.punctuation)):
            num_tokens += 1
            length += len(token)
    if body != "":
        feats[15] = length/num_tokens if num_tokens != 0 else 0
    
    # feats[16] : Number of sentences
    feats[16] = len(sentences)

    feats = feats17_28(body, feats)
    return feats

def feats17_28(body, feats):

    # feats[17] : Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # feats[18] : Average of IMG from Bristol, Gilhooly, and Logie norms
    # feats[19] : Average of FAM from Bristol, Gilhooly, and Logie norms
    # feats[20] : Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    # feats[21] : Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    # feats[22] : Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    # feats[24] : Average of V.Mean.Sum from Warringer norms
    # feats[25] : Average of A.Mean.Sum from Warringer norms
    # feats[26] : Average of D.Mean.Sum from Warringer norms
    # feats[27] : Standard deviation of V.Mean.Sum from Warringer norms
    # feats[28] : Standard deviation of A.Mean.Sum from Warringer norms
    
    AoA_list, IMG_list, FAM_list = [], [], []
    VMS_list, AMS_list, DMS_list = [], [], []
    for i in body:
        if i in bgl_data:
            AoA, IMG, FAM = [float((val if val != "" else 0)) for val in bgl_data[i]]
            AoA_list.append(AoA)
            IMG_list.append(IMG)
            FAM_list.append(FAM)
        if i in warr_data:
            VMS, AMS, DMS = [float((val if val != "" else 0)) for val in warr_data[i]]
            VMS_list.append(VMS)
            AMS_list.append(AMS)
            DMS_list.append(DMS)
    AoA_list = np.array(AoA_list); IMG_list = np.array(IMG_list); FAM_list = np.array(FAM_list)
    VMS_list = np.array(VMS_list); AMS_list = np.array(AMS_list); DMS_list = np.array(DMS_list)

    if len(AoA_list) >0:
        feats[17] = np.mean(AoA_list)
        feats[20] = np.std(AoA_list)

    if len(IMG_list) >0:
        feats[18] = np.mean(IMG_list)
        feats[21] = np.std(IMG_list)

    if len(FAM_list) >0:
        feats[19] = np.mean(FAM_list)
        feats[22] = np.std(FAM_list)

    if len(VMS_list) >0:
        feats[23] = np.mean(VMS_list)
        feats[26] = np.std(VMS_list)

    if len(AMS_list) >0:
        feats[24] = np.mean(AMS_list)
        feats[27] = np.std(AMS_list)

    if len(DMS_list) >0:
        feats[25] = np.mean(DMS_list)
        feats[28] = np.std(DMS_list)
    return feats

    
def extract2(feat, comment_class, comment_id):
    ''' This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this 
        function adds feature 30-173). This should be a modified version of 
        the parameter feats.
    '''    
    feat[29:] = file_data[comment_class][comment_id]
    return feat


def main(args):
    #Declare necessary global variables here. 
    #Load data
    data = json.load(open(args.input))
    
    feats = np.zeros((len(data), 173+1))

    for (i, comment) in enumerate(data):
        
        # TODO: Call extract1 for each datatpoint to find the first 29 features. 
        # Add these to feats.
        # fill the first 29 features 

        # for testing purposes only ** remove later **
        # comment=data[7]
        features = extract1(comment['body'])

        # TODO: Call extract2 for each feature vector to copy LIWC features (features 30-173)
        # into feats. (Note that these rely on each data point's class,
        # which is why we can't add them in extract1).
        # fill features 29-173
        features = extract2(features, comment['cat'], comment['id'])
        
        feats[i, :-1] = features
        # 174th is the category (label)
        feats[i, -1] = categories.index(comment['cat'])

        if (i + 1) % 1000 == 0:
            print(f"iter: '{i + 1}'")


    # save output as a zipped numpy file
    np.savez_compressed(args.output, feats)

    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        
    
    main(args)

