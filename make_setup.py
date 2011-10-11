"""
Create setup files used to train models
"""

import os, sys, re
import util

def fix_wsj_trans(s):
    s = s.replace('\.', '.')
    s = s.replace('.PERIOD', '\.PERIOD')
    s = s.replace('.POINT', '\.POINT')
    s = s.replace('...ELLIPSIS', '\...ELLIPSIS')
    s = re.sub("([^ ])\\\\'", "\\1'", s)
    s = re.sub('([^ ])\\\-', '\\1-', s)
    s = s.replace('--DASH', '-DASH')
    s = re.sub('([A-Za-z])\:([A-Za-z]*)', '\\1\\2', s)
    s = s.replace('*', '')
    s = s.replace(' !', ' ')
    s = s.replace('~', '')
    return s.strip()

def wsj(path_wav, path_trans, config, output, wav_list=[]):
    
    ## wav files
    wav_files = set(util.get_files(path_wav, re.compile('.*\.wv1', re.IGNORECASE)))
    print 'found wav files [%d]' %len(wav_files)

    ## filter using a wav list
    keys = set()
    if wav_list:
        hint = path_wav.split('/')[-2]+'/'
        files = os.popen('grep -i %s %s' %(hint, wav_list)).read().splitlines()
        if len(files) == 0:
            files = [os.path.basename(f).split('.')[0] for f in open(wav_list).read().splitlines()]
        for file in files:
            key = os.path.basename(file).split('.')[0].upper()
            keys.add(key)
        print 'found keys [%d]' %len(list(keys))
    fh = open(output, 'w')

    ## transcription files
    trans_files = util.get_files(path_trans, re.compile('.*\.dot', re.IGNORECASE))
    print 'found transcription files [%d]' %len(trans_files)
    unmatched = 0
    for file in trans_files:
        dirname = os.path.dirname(file)
        for line in open(file):
            line = line.strip()
            ext = re.search('\([^()]*\)$', line).group()
            trans = line.replace(ext, '').strip()
            trans = fix_wsj_trans(trans)
            ext = ext.replace('(','').replace(')','').upper()
            wav_file = '%s/%s.WV1' %(dirname, ext)
            if wav_file not in wav_files: wav_file = '%s/%s.wv1' %(dirname, ext.lower())
            if wav_file not in wav_files:
                for w in wav_files:
                    if ext in w or ext.lower() in w: wav_file = w
            if wav_file not in wav_files:
                print 'no matching wav file [%s]' %wav_file
                unmatched += 1
                continue

            if ext in keys or len(keys)==0:
                fh.write('%s %s %s\n' %(wav_file, config, trans))
    fh.close()
    print 'unmatched [%d]' %unmatched


def fix_swb_trans(words):
    cleaned = []
    for word in words:
        if word.startswith('[') and word.endswith(']'): continue
        elif '-' in word:
            items = word.split('-')
            if len(items) == 2:
                if items[0] in ['TWENTY', 'THIRTY', 'FORTY', 'FIFTY', 'SIXTY', 'SEVENTY', 'EIGHTY', 'NINETY'] and items[1] in ['ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']:
                    cleaned.extend(items)
        else: cleaned.append(word)
    return ' '.join(cleaned).strip()
    
def swboard(path_wav, path_trans, config, output, wav_list=[]):
    """
    <wav id> <channelId> <speakerId> <begin time segment> <end time segment> <label> text
    fsh_60262 1 fsh_60262_A 47.9  49.77 <O,FI,M,STANDARD>  I DO IT IS HALLOWEEN
    """
    
    ## wav files
    #wav_files = set(util.get_files(path_wav, re.compile('.*\.wav', re.IGNORECASE)))
    #print 'found wav files [%d]' %len(wav_files)

    ## load speaker map
    speaker_by_conv = {}
    convs_by_speaker = {}
    for line in open('common/swb_speaker.map'):
        [conv, speaker] = line.split()
        speaker_by_conv[conv] = speaker
        if not speaker in convs_by_speaker: convs_by_speaker[speaker] = []
        convs_by_speaker[speaker].append(conv)
    sorted_speakers = convs_by_speaker.items()
    sorted_speakers.sort(lambda x,y: len(y[1]) - len(x[1]))

    ## split convs into sets
    train1, train2, test = set(), set(), set()
    history = [1]
    count1, count2 = 0, 0
    for speaker, convs in sorted_speakers:
        print speaker, convs
        if len(convs) == 1: test.add(convs[0])
        else:
            if history[-2:] == [1,1] or history[-2:] == [1,2]:
                train2.update(set(convs))
                history.append(2)
                count2 += 1
            else:
                train1.update(set(convs))
                history.append(1)
                count1 += 1
    #print len(train1), len(train2), len(test)
    #print count1, count2
    #sys.exit()
    
    ## transcription files
    trans_files = util.get_files(path_trans, re.compile('.*-trans\.text', re.IGNORECASE))
    print 'found transcription files [%d]' %len(trans_files)

    output1 = output + '_1'
    output2 = output + '_2'
    output3 = output + '_3'
    fh1, fh2, fh3 = open(output1, 'w'), open(output2, 'w'), open(output3, 'w')
    stm1, stm2, stm3 = open(output1 + '.stm', 'w'), open(output2 + '.stm', 'w'), open(output3 + '.stm', 'w')
    
    utt_count = 0
    file_count = 0
    token_count = 0
    word_list = set()
    for file in trans_files:
        file_count += 1
        if file_count % 100 == 0: print 'transcription files processed [%d]' %file_count
        dirname = os.path.dirname(file)
        for line in open(file):
            items = line.strip().split()
            id = items[0]
            [id, ms98, a, utt] = id.split('-')
            side = id[-1]
            id = id[:-1]
            start, end = items[1], items[2]
            words = map(str.upper, items[3:])
            dir = '%s_%s' %(id, side)
            new_id = '%s%s-ms98-a-%s' %(id, side, utt)
            wav_file = path_wav + '%s/%s.wav' %(dir, new_id)
            if not os.path.isfile(wav_file):
                continue
            if sum([int(w.startswith('[') or w.endswith(']')) for w in words]) == len(words):
                continue
            if len(words) == 0: continue

            trans = fix_swb_trans(words)
            if len(trans) == 0: continue
            if trans.isdigit(): continue

            conv = (id + '_' + side).replace('sw', '')
            if conv in train1:
                fh = fh1
                stm = stm1
            elif conv in train2:
                fh = fh2
                stm = stm2
            elif conv in test:
                fh = fh3
                stm = stm3
            else:
                print 'No set exists for conv: %s' %conv
                print trans
                continue
            
            fh.write('%s %s %s\n' %(wav_file, config, trans))
            stm.write('%s %s %s%s %s %s %s\n' %(new_id, side, id, side, start, end, trans))
            utt_count += 1
            for word in words: word_list.add(word)
            token_count += len(words)

    fh.close()
    stm.close()
    print 'Using [%s] wrote [%d] utts to [%s]' %(path_trans, utt_count, output)
    print 'Tokens [%d] Types [%d]' %(token_count, len(word_list))

def fisher(path_wav, path_trans, output):

    ## transcription files
    trans_files = util.get_files(path_trans, re.compile('\.txt', re.IGNORECASE))
    trans_files = [f for f in trans_files if 'bbn' not in f]
    #print 'found transcription files [%d]' %len(trans_files)

    fh = open(output, 'w')

    file_count = 0
    token_count = 0
    word_list = set()
    for file in trans_files:
        file_count += 1
        if file_count % 100 == 0: print 'transcription files processed [%d]' %file_count
        id = os.path.basename(file).replace('.txt', '')
        for line in open(file):
            line = line.strip()
            if not line: continue
            if line.startswith('#'): continue
            words = map(str.upper, ' '.join(line.split(':')[1:]).split())
            if not words: continue
            fh.write(' '.join(words) + '\n')
    fh.close()

def timit(path_wav, path_trans, output, config, test_speakers=None):

    ## transcription files
    trans_files = [f for f in util.get_files(path_trans, re.compile('\.txt', re.IGNORECASE)) if '/si' in f or '/sx' in f]
    if test_speakers != None:
        trans_files = [f for f in trans_files if f.split('/')[-2][1:] in test_speakers]
    print 'found transcription files [%d]' %len(trans_files)

    fh = open(output, 'w')
    for trans_file in trans_files:
        wav_file = trans_file.replace('.txt', '.wav')
        if not os.path.isfile(wav_file): continue
        trans = open(trans_file).read().lower()
        trans = ' '.join(trans.split()[2:])
        trans = re.sub('[^a-z ]', '', trans)
        fh.write('%s %s %s\n' %(wav_file, config, trans))
    fh.close()
            
if __name__ == '__main__':

    wsj0_train = '/u/dgillick/workspace/hmm/data/wsj0_data/SI_TR_S/'
    wsj0_nov92 = '/u/dgillick/workspace/hmm/data/wsj0_data/SI_ET_05/'
    wsj1_train = '/u/dgillick/workspace/hmm/data/wsj1_data/si_tr_s/'
    wsj1_sidt05 = '/u/dgillick/workspace/hmm/data/wsj1_data/si_dt_05/'

    si84_list = '/u/dgillick/workspace/hmm/data/wsj0_data/WSJ0/DOC/INDICES/TRAIN/TR_S_WV1.NDX'
    nov92_list = '/u/dgillick/workspace/hmm/data/wsj0_data/WSJ0/DOC/INDICES/TEST/NVP/SI_ET_05.NDX'
    si200_list = '/u/dgillick/workspace/hmm/data/wsj1_data/doc/indices/si_tr_s.ndx'
    sidt05_list = '/u/dgillick/workspace/hmm/htk_recipe/common/si_dt_05_odd.ndx'
    
    wsj_config = 'wsj.htk_config'

    # wsj0 setup
    wsj(wsj0_train, wsj0_train, wsj_config, 'si84.setup', si84_list)

    ## nov92 setup
    wsj(wsj0_nov92, wsj0_nov92, wsj_config, 'nov92.setup', nov92_list)

    ## wsj1 setup
    wsj(wsj1_train, wsj1_train, wsj_config, 'si200.setup', si200_list)

    ## si_dt_05_odd setup
    wsj(wsj1_sidt05, wsj1_sidt05, wsj_config, 'si_dt_05_odd.setup', sidt05_list)

    sys.exit()

    #----------------------------------------------------------#

    swboard_wav = '/u/drspeech/data/swboard/SWB1-seg/segmented/waveforms/'
    swboard_ms = '/u/drspeech/data/swboard/SWB1-seg/isip-data/swb_ms98_transcriptions/'
    swboard_config = 'swboard.htk_config'

    #swboard(swboard_wav, swboard_ms, swboard_config, 'swboard.setup')
    #sys.exit()

    #----------------------------------------------------------#

    fisher_wav = '/u/drspeech/data/fisher/segmented/waveforms/'
    fisher_trans = '/u/drspeech/data/fisher/LDC/'

    #fisher(fisher_wav, fisher_trans, 'fisher.txt')

    timit_wav = '/u/drspeech/data/timit/dist/'
    timit_trans = '/u/drspeech/data/timit/dist/'
    timit_test_speakers = ['dab0', 'wbt0', 'elc0', 'wew0', 'pas0', 'jmp0', 'lnt0', 'pkt0', 'lll0', 'tls0', 'jlm0', 'bpm0', 'klt0', 'nlp0', 'cmj0', 'jdh0', 'mgd0', 'grt0', 'njm0', 'dhc0', 'jln0', 'pam0', 'mld0', 'tas1']

    timit(timit_wav+'train/', timit_trans+'train/', 'timit_train.setup', wsj_config)
    timit(timit_wav+'test/', timit_trans+'test/', 'timit_test.setup', wsj_config, timit_test_speakers)
