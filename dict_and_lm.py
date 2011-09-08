"""
Functions for building dictionaries and language models
"""

import os, re, gzip
import util
import coding

def make_mlf_from_transcripts(model, orig_dict, setup, data_path, word_mlf, mfc_list, skip_oov=True):
    """
    An MLF is an HTK-formatted transcription file. This is created
    from the word-level transcripts in setup.
    """
    
    replace_escaped_words = True

    ## Load the dictionary words
    dict_words = set([entry.split()[0].upper() for entry in open(orig_dict).read().splitlines()
                      if not entry.startswith('#') and len(entry.strip()) > 0])
    words = set()

    if setup.endswith('gz'): setup_reader = lambda x: gzip.open(x)
    else: setup_reader = lambda x: open(x)

    ## Create MLF-format entries for each utterance
    mfcs = []
    mlf = ['#!MLF!#']
    count = 0
    for line in setup_reader(setup):
        skip = False
        items = line.strip().split()
        wav = items[0]
        mfc = coding.get_mfc_name_from_wav(wav, data_path)
        curr = ['"*/%s.lab"' %os.path.basename(wav).split('.')[0]]
        trans = map(str.upper, items[2:])
        for word in trans:
            if replace_escaped_words and '\\' in word:
                new_word = re.sub(r'\\[^A-Za-z]*', r'', word)
                if new_word in dict_words: word = new_word
                
            if word not in dict_words:
                ## Don't include bracketed words or periods in the labels
                if word.startswith('[') and word.endswith(']'): continue
                if word == '.': continue
                if model.verbose > 1: util.log_write(model.logfh, '  Not in dictionary [%s]' %word)
                
                ## Remove the utterance if there are other non-dictionary words
                if skip_oov: skip = True

            if word[0].isdigit(): word = '_' + word
            curr.append(word)

        ## Check for empty transcriptions
        if len(curr) <= 1: skip = True

        curr.append('.')
        if not skip:
            mlf.extend(curr)
            for word in curr:
                words.add(word)
            mfcs.append(mfc)
            count += 1

    ## Write the MLF
    fh = open(word_mlf, 'w')
    fh.write('\n'.join(mlf) + '\n')
    fh.close()
    
    ## Create a new MFC list file
    fh = open(mfc_list, 'w')
    for mfc in mfcs: fh.write('%s\n' %mfc)
    fh.close()
    
    return count, words

def make_decode_dict(dict, decode_dict, words):
    """
    Make a decoding dictionary that also includes <s> and </s>
    """
    
    fh2 = open(decode_dict, 'w')
    fh2.write('<s> sil\n')
    fh2.write('</s> sil\n')
    count = 0
    for line in open(dict):
        line = line.strip()
        if len(line) <= 0: continue
        if line.startswith('#'): continue
        clean_line = re.sub(r'\(\d\)', r'', line)
        word = clean_line.split()[0].upper()
        if word[0].isdigit(): word = '_' + word
        if not word in words: continue
        pron = ' '.join(clean_line.split()[1:])
        count += 1
        fh2.write('%s\t\t%s\n' %(word, pron))

    fh2.close()
    return count

def make_train_dict(dict, train_dict, words):
    """
    Make a training dictionary with all the words in the training set
    """
    
    fh1 = open(train_dict, 'w')
    count = 0
    for line in open(dict):
        line = line.strip()
        if len(line) <= 0: continue
        if line.startswith('#'): continue
        clean_line = re.sub(r'\(\d\)', r'', line)
        word = clean_line.split()[0].upper()
        if word[0].isdigit(): word = '_' + word
        if not word in words: continue
        pron = ' '.join(clean_line.split()[1:])
        count += 1
        fh1.write('%s\t\t%s sp\n' %(word, pron))
        fh1.write('%s\t\t%s sil\n' %(word, pron))

    fh1.write('silence sil\n')
    fh1.close()
    return count

def build_lm_from_mlf(model, word_mlf, dictionary, vocab, lm_dir, lm, lm_order, target_ppl=None):
    """
    Build a language model using SRILM
    Use the transcripts in the word mlf
    Output to lm
    Output intermediate files in lm_dir
    Return perplexity on the training text
    """

    dict = set([entry.split()[0].upper() for entry in open(dictionary).read().splitlines()
                if not entry.startswith('#') and len(entry.strip()) > 0])

    ## Prepare to build an LM by creating a file with one sentence per line
    text_file = '%s/training.txt' %lm_dir
    text, curr = [], []
    
    ## Extract a vocab from the MLF
    cmd = 'cat %s | grep ".lab" -v | grep "MLF" -v | sort | uniq' %word_mlf
    mlf_vocab = set(os.popen(cmd).read().splitlines())
    mlf_dict_vocab = list(mlf_vocab.intersection(dict))
    mlf_dict_vocab.sort()
    fh = open(vocab, 'w')
    for word in mlf_dict_vocab: fh.write(word + '\n')
    fh.close()

    for line in open(word_mlf):
        line = line.strip()
        if line.startswith('#!MLF'): continue
        if line.startswith('"') and '.lab' in line: continue
        if line == '.':
            text.append(' '.join(curr))
            curr = []
            continue
        curr.append(line)

    fh = open(text_file, 'w')
    fh.write('\n'.join(text))
    fh.close()

    ## Build a language model
    cutoff, cutoff_min, cutoff_max = 5, 1, 50
    iters, prev_cutoff = 0, 0

    cmd = 'ngram-count -vocab %s -order %d -text %s -lm %s' %(vocab, lm_order, text_file, lm)
    util.run(cmd, lm_dir)
    cmd = 'ngram -order %d -lm %s -ppl %s -debug 0' %(lm_order, lm, text_file)
    res = util.run(cmd, lm_dir)
    ppl = float(os.popen('grep zeroprobs %s' %res).read().split()[5])
    if not target_ppl_ratio: return ppl
    util.log_write(model.logfh, '  cutoff [%d] gives ppl [%1.2f]' %(1, ppl))
    target_ppl = ppl * target_ppl_ratio

    while True:
        iters += 1
        params = '-gt%dmin %d' %(lm_order, cutoff)
        cmd = 'ngram-count -vocab %s -order %d -text %s -lm %s %s' %(vocab, lm_order, text_file, lm, params)
        util.run(cmd, lm_dir)
        cmd = 'ngram -order %d -lm %s -ppl %s -debug 0' %(lm_order, lm, text_file)
        res = util.run(cmd, lm_dir)
        ppl = float(os.popen('grep zeroprobs %s' %res).read().split()[5])

        if not target_ppl or abs(ppl - target_ppl) < 1: break
        if cutoff == prev_cutoff or iters > 10: break
        prev_cutoff = cutoff
        util.log_write(model.logfh, '  cutoff [%d] gives ppl [%1.2f]' %(cutoff, ppl))

        if ppl < target_ppl:
            cutoff_min = cutoff
            cutoff = (cutoff + cutoff_max) / 2
        else:
            cutoff_max = cutoff
            cutoff = (cutoff + cutoff_min) / 2

    ## Return perplexity on the training data
    return ppl
