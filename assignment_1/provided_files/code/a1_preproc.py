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

import sys
import argparse
import os
import json
import re 
import spacy #3.2.1
import html

# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
# spacy sentencizer
nlp.add_pipe('sentencizer')


def preproc1(comment , steps=range(1, 6)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    modComm = comment

    if 1 in steps:  
        #modify this to handle other whitespace chars.
        # replace newlines with spaces  
        modComm = re.sub(r"\n{1,}", " ", modComm)
        # replace any missed new lines, tabs, carriage returns, formfeed, vertical tabs with spaces
        modComm = re.sub(r"\n+|\t+|\r+|\f+|\v+", " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        modComm = re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", modComm)
        
    if 4 in steps: #remove duplicate spaces.
        #1. remove trailing and leading spaces
        modComm = re.sub(r"^\s+|\s+$", "", modComm)
        #2. remove duplicate spaces
        modComm = re.sub(r"\s+", " ", modComm)

    if 5 in steps:
        # TODO: get Spacy document for modComm (-)
        utt = nlp(modComm)
        # variable to store processed text
        processedLine = ""
        # TODO: use Spacy document for modComm to create a string.
        # Make sure to:
        #    * Split tokens with spaces. (-)
        #    * Write "/POS" after each token. (-)
        #    * Insert "\n" between sentences. (-)

        for sentence in utt.sents:
            tokens = []
            for token in sentence:
                # token original text
                text = token.text
                # token tag
                tag = token.tag_
                # token lemma
                lemma = token.lemma_
                # variable to store tagged text 
                token = lemma
                # case: if lemma begins with a dash (‘-’) when the token doesn’t, replace token with text of token; otherwise keep lemma
                if lemma.startswith("-"):
                    if (not text.startswith("-")):
                        token = text
                
                # tag each sentence with POS e.g. dog/NN
                    # retain the case of the original token
                        # case: if the original token is entirely in uppercase, then so is the lemma; 
                            #   otherwise, keep the lemma in lowercase.
                if text.isupper():
                    # upper case lemma when originally stored text is upper
                    posTaggedToken = token.upper() + '/' + tag
                else:
                    # lower case lemma otherwise 
                    posTaggedToken = token.lower() + '/' + tag

                # append posTaggedToken to the list of all tokens 
                tokens.append(posTaggedToken)

            # insert "\n" between processed sentences, where is a sentence is formed by joining pos tagged tokens by a " "
            processedLine += " ".join(tokens) + '\n'
        modComm = processedLine
    
    return modComm


def main(args):
    allOutput = []
    
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            
            # TODO: select appropriate args.max lines ()
            # end_index - start_index = 10,000 (processing 10,000 per file)
            
            start_index = int(args.ID[0] % len(data))
            end_index =  int(start_index + args.max)  #here the max can be set to a diff value in args 
            
            for i in range(start_index, end_index, 1):
                
                # use circular indexing to avoid problems that might arise when the student number is too
                # close to the end (len(data) = 599872)
                i %= len(data) # e.g. if i < len(data) i=i, if not then we begin incrementing from the beginning

                # TODO: read those lines with something like `j = json.loads(line)` (-)
                comment = json.loads(data[i])

                # TODO: choose to retain fields from those lines that are relevant to you (-)
                comment_fields = {} # dict
                comment_fields['id'] = comment['id']
                comment_fields['body'] = comment['body']
                
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)  (-)
                comment_fields['cat'] = file
                
                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument (-)
                processed_body = preproc1(comment_fields['body'])

                # TODO: replace the 'body' field with the processed text (-)
                comment_fields['body'] = processed_body

                # TODO: append the result to 'allOutput' (-)
                allOutput.append(comment_fields)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
