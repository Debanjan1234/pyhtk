"""
Functions for initializing HMMs for training
"""

import os, random, re
import util
random.seed(0)

def word_to_phone_mlf(model, dict, word_mlf, phone_mlf, mono_list):
    """
    Convert the word-level mlf to a phone level mlf with HLEd
    """

    if not os.path.isfile(word_mlf):
        util.log_write(model.logfh, 'No word MLF file here [%s]' %word_mlf)
        util.exit(model.log)

    if not os.path.isfile(dict):
        util.log_write(model.logfh, 'No dict file here [%s]' %dict)
        util.exit(model.log)

    ## Create mkphones0.led
    led_file = '%s/mkphones0.led' %model.exp
    fh = open(led_file, 'w')
    fh.write('EX\nIS sil sil\n')
    fh.close()

    ## Convert the word level MLF into a phone MLF
    cmd_log = '%s/hhed_word_to_phone.log' %model.exp
    cmd  = 'HLEd -A -T 1 -l "*"'
    cmd += ' -d %s' %dict
    cmd += ' -i %s' %phone_mlf
    cmd += ' %s %s > %s' %(led_file, word_mlf, cmd_log)
    os.system(cmd)

    ## Create list of phones (appearing in the phone MLF)
    monophones = set()
    for line in open(phone_mlf):
        phone = line.strip()
        if phone.isalpha(): monophones.add(phone)
    monophones = list(monophones)
    monophones.sort()
    fh = open(mono_list, 'w')
    for phone in monophones: fh.write('%s\n' %phone)
    fh.close()

    return len(monophones)

def make_proto_hmm(model, mfc_list, proto_hmm):
    """
    Create a prototype HMM based on:
    - number of features
    - number of states
    """

    ## Start writing a proto hmm
    fh = open(proto_hmm, 'w')

    ## Get info from feature files
    lines = os.popen('HList -z -h -r %s' %open(mfc_list).read().splitlines()[0]).read().splitlines()
    kind = lines[0].split()[-1].replace('_C_K', '')
    comps = int(lines[1].split()[2])

    fh.write('~o <VecSize> %d <%s>\n' %(comps, kind))
    fh.write('~h "proto_hmm"\n\n')
    fh.write('<BeginHMM>\n')
    fh.write('\t<NumStates> %d\n' %model.states)
    for index in range(2, model.states):
        fh.write('\t<State> %d\n' %index)
        fh.write('\t\t<Mean> %d\n' %comps)
        fh.write('\t\t\t%s\n' %' '.join(['0.0' for i in range(comps)]))
        fh.write('\t\t<Variance> %d\n' %comps)
        fh.write('\t\t\t%s\n' %' '.join(['1.0' for i in range(comps)]))
    fh.write('\t<TransP> %d\n' %model.states)
    fh.write('\t\t%s\n' %' '.join(['%1.1f' %float(i==1) for i in range(model.states)]))
    for i in range(2, model.states):
        fh.write('\t\t%s\n' %' '.join(['%1.1f' %(float(j in [i-1,i])/2.) for j in range(model.states)]))
    fh.write('\t\t%s\n' %' '.join(['0.0' for i in range(model.states)]))
    fh.write('<EndHMM>\n')
    fh.close()

def initialize_hmms(model, root_dir, mfc_list, mono_list, proto_hmm):
    """
    Compute mean and variance of each feature across all utterances and
    set all Gaussians in the prototype HMM to have the same mean and variance
    """

    output_dir = '%s/HMM-0-0' %root_dir
    util.create_new_dir(root_dir)
    util.create_new_dir(output_dir)
    cmd_log = '%s/hcompv.log' %output_dir

    ## Sample from the full mfc list to reduce computation
    sampled_mfc_list = '%s/mfc_sample.list' %output_dir
    fh = open(sampled_mfc_list, 'w')
    mfcs = open(mfc_list).read().splitlines()
    random.shuffle(mfcs)
    mfc_frac = model.var_floor_fraction
    num_mfcs_for_hcompv = int(mfc_frac * len(mfcs))
    for mfc in mfcs[:num_mfcs_for_hcompv]:
        fh.write('%s\n' %mfc)
    fh.close()

    cmd  = 'HCompV -A -T 1 -m'
    cmd += ' -C %s' %model.mfc_config
    cmd += ' -f 0.01'
    cmd += ' -S %s' %sampled_mfc_list
    cmd += ' -M %s' %output_dir
    cmd += ' %s > %s' %(proto_hmm, cmd_log)

    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    ## Copy the initial HMM for each monophone
    proto_hmm = '%s/proto_hmm' %output_dir    
    hmm_defs_init = '%s/init.mmf' %output_dir
    hmm = re.search(re.compile('<BEGINHMM>.*<ENDHMM>', re.DOTALL), open(proto_hmm).read()).group()

    fh = open(hmm_defs_init, 'w')
    for line in open(mono_list):
        phone = line.strip()
        fh.write('~h "%s"\n' %phone)
        fh.write('%s\n' %hmm)
    fh.close()

    ## Create mmf header file
    ## TODO: get rid of this?
    mmf_header = '%s/header.mmf' %output_dir
    cmd = 'head -3 %s | cat - %s/vFloors > %s' %(proto_hmm, output_dir, mmf_header)
    os.system(cmd)

    ## Fix sp and silence models
    cleanup_config = '%s/cleanup_init.hed' %output_dir
    fh = open(cleanup_config, 'w')
    fh.write('AT 4 2 0.2 {sil.transP}\n')
    fh.write('AT 2 4 0.2 {sil.transP}\n')
    fh.write('AT 1 5 0.3 {sp.transP}\n')
    fh.write('TI silsp_2 {sil.state[2],sp.state[2]}\n')
    fh.write('TI silsp_3 {sil.state[3],sp.state[3]}\n')
    fh.write('TI silsp_4 {sil.state[4],sp.state[4]}\n')
    fh.close()
    
    hmm_defs_final = '%s/MMF' %output_dir
    cmd_log = '%s/hhed_sil.log' %output_dir
    cmd  = 'HHEd -A -D -T 1 -d %s' %output_dir
    cmd += ' -H %s -H %s' %(mmf_header, hmm_defs_init)
    cmd += ' -M %s' %output_dir
    cmd += ' -w %s' %hmm_defs_final
    cmd += ' %s %s > %s' %(cleanup_config, mono_list, cmd_log)
    os.system(cmd)

    return output_dir, num_mfcs_for_hcompv
