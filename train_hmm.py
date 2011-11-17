"""
Functions for training HMMs: forward-backward, alignments, state-tying, and mixing up
"""

import os, sys
import util

HEREST_CMD = 'HERest'
#HEREST_CMD = '/u/arlo/bin/fast_htk/v0/HERest'

def run_iter(model, root_dir, prev_dir, mlf_file, model_list, mix_size, iter, extra):
    """
    Run an iteration of Baum-Welch training using HERest
    """

    output_dir = '%s/HMM-%d-%d' %(root_dir, mix_size, iter)
    util.create_new_dir(output_dir)

    mfc_list = '%s/mfc.list' %model.exp
    utts_per_split = max(250, (1 + (model.setup_length / 200)))

    ## HERest parameters
    min_train_examples = 0
    prune_thresh = 250
    prune_inc = 150
    prune_limit = 2000

    def herest(input, split_num, extra):
        try: log_id = os.path.basename(input).split('.')[2]
        except: log_id = 'acc'
        cmd  = '%s -D -A -T 1 -m %d' %(HEREST_CMD, min_train_examples)
        cmd += ' -t %d %d %d' %(prune_thresh, prune_inc, prune_limit)
        cmd += ' -s %s/stats' %output_dir
        cmd += ' -C %s%s' %(model.mfc_config, extra)
        cmd += ' -I %s' %mlf_file
        cmd += ' -H %s/MMF' %prev_dir
        cmd += ' -p %d' %split_num
        cmd += ' -S %s' %input
        #cmd += ' -M %s %s' %(output_dir, model_list)
        cmd += ' -M %s %s >> %s/herest.%s.log' %(output_dir, model_list, output_dir, log_id)
        return cmd

    ## Split up MFC list with unix split
    cmd = 'split -a 4 -d -l %d %s %s/%s' %(utts_per_split, mfc_list, output_dir, 'mfc.list.')
    os.system(cmd)

    ## Create the HERest commands
    cmds = []
    inputs = os.popen('ls %s/mfc.list.*' %output_dir).read().splitlines()
    split_num = 0
    for input in inputs:
        split_num += 1
        cmds.append(herest(input, split_num, extra))

    ## Non-parallel case
    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd)

    ## Parallel case: one command per line in cmds_file
    else:
        cmds_file = '%s/herest.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

    ## Gather the created .acc files
    acc_file = '%s/herest.list' %output_dir
    os.system('ls %s/HER*.acc > %s' %(output_dir, acc_file))

    ## Combine acc files into a new HMM
    cmd = herest(acc_file, 0, extra)
    cmd = cmd.split('>>')[0]
    cmd += ' >> %s/herest.log' %output_dir
    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    ## Clean up
    os.system('rm -f %s/mfc.list.* %s/HER*.acc' %(output_dir, output_dir))
    os.system('bzip2 %s/herest.*.log %s/run-command*.log' %(output_dir, output_dir))

    ## Get a few stats
    num_models = int(os.popen('grep "<MEAN>" %s/MMF -c' %output_dir).read().strip())
    likelihood = float(os.popen('cat %s/herest.log | grep aver' %output_dir).read().strip().split()[-1])

    return output_dir, num_models, likelihood

def mixup(model, root_dir, prev_dir, model_list, mix_size, estimateVarFloor=0):
    """
    Run HHEd to initialize a mixup to mix_size gaussians
    """

    output_dir = '%s/HMM-%d-%d' %(root_dir, mix_size, 0)
    util.create_new_dir(output_dir)

    ## Make the hed script
    mix_hed = '%s/mix_%d.hed' %(output_dir, mix_size)
    fh = open(mix_hed, 'w')

    if estimateVarFloor:
            fh.write('LS %s/stats\n' %prev_dir)
            fh.write('FA 0.1\n')
            
    fh.write('MU %d {(sil,sp).state[2-%d].mix}\n' %(2*mix_size,model.states-1))
    fh.write('MU %d {*.state[2-%d].mix}\n' %(mix_size, model.states-1))
    fh.close()

    hhed_log = '%s/hhed_mix.log' %output_dir

    cmd  = 'HHEd -A -D -T 1 -H %s/MMF -M %s' %(prev_dir, output_dir)
    cmd += ' %s %s > %s' %(mix_hed, model_list, hhed_log)
    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    return output_dir


def mixdown_mono(model, root_dir, prev_dir, phone_list):
    """
    Run HHEd to mixdown monophones
    """

    output_dir = '%s/HMM-1-0' %root_dir
    util.create_new_dir(output_dir)

    ## Create the full list of possible triphones
    phones = open(phone_list).read().splitlines()
    non_sil_phones = [p for p in phones if p not in ['sp', 'sil']]

    ## Make the hed script
    mixdown_hed = '%s/mix_down.hed' %output_dir
    fh = open(mixdown_hed, 'w')
    fh.write('MD 12 {(sil,sp).state[2-%d].mix}\n' %(model.states-1))
    for phone in non_sil_phones:
        fh.write('MD 1 {%s.state[2-%d].mix}\n' %(phone, model.states-1))
    fh.close()

    hhed_log = '%s/hhed_mixdown.log' %output_dir

    cmd  = 'HHEd -A -D -T 1 -H %s/MMF -M %s' %(prev_dir, output_dir)
    cmd += ' %s %s > %s' %(mixdown_hed, phone_list, hhed_log)
    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    return output_dir


def align(model, root_dir, mfc_list, model_dir, word_mlf, new_mlf, model_list, dict, align_config): 
    """
    Create a new alignment based on a model and the word alignment with HVite
    """

    output_dir = '%s/Align' %root_dir
    util.create_new_dir(output_dir)
    utts_per_split = max(100, (1 + (model.setup_length / 200)))

    ## Copy old mfc list
    os.system('cp %s %s/mfc_old.list' %(mfc_list, output_dir))

    ## HVite parameters
    prune_thresh = 250

    def hvite(input, output):
        #-o SWT 
        cmd  = 'HVite -D -A -T 1 -b silence -a -m -y lab '
        cmd += '-t %d' %prune_thresh
        cmd += ' -C %s' %align_config
        cmd += ' -H %s/MMF' %model_dir
        cmd += ' -i %s' %output
        cmd += ' -I %s' %word_mlf
        cmd += ' -S %s' %input
        cmd += ' %s %s' %(dict, model_list)
        cmd += ' >> %s.hvite.log' %output
        return cmd

    ## Split up MFC list with unix split
    cmd = 'split -a 4 -d -l %d %s %s/%s' %(utts_per_split, mfc_list, output_dir, 'mfc.list.')
    os.system(cmd)

    ## Create the HVite commands
    cmds = []
    outputs = []
    inputs = os.popen('ls %s/mfc.list.*' %output_dir).read().splitlines()
    for input in inputs:
        output = input.replace('mfc.list', 'align.output')
        outputs.append(output)
        cmds.append(hvite(input, output))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hvite.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

    ## Merge and fix silences
    ## TODO: -s file_list
    merge_sil = '%s/merge_sp_sil.led' %output_dir
    fh = open(merge_sil, 'w')
    fh.write('ME sil sp sil\n')
    fh.write('ME sil sil sil\n')
    fh.write('ME sp sil sil\n')
    fh.close()

    cmd = 'HLEd -D -A -T 1 -i %s %s %s >> %s/hled.log' %(new_mlf, merge_sil, ' '.join(outputs), output_dir)
            
    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    ## Prune failed alignments from the mfc list
    bad_count = 0
    mlf_labels = os.popen('grep "\.lab" %s' %new_mlf).read().splitlines()
    mlf_labels = set([os.path.basename(s).split('.')[0] for s in mlf_labels])
    mfc_labels = open(mfc_list).read().splitlines()
    fh = open(mfc_list, 'w')
    for mfc in mfc_labels:
        id = os.path.basename(mfc).split('.')[0]

        ## Check for missing transcriptions
        if id not in mlf_labels:
            if model.verbose > 0: util.log_write(model.logfh, 'removed bad alignment [%s]' %id)
            bad_count += 1
        else: fh.write(mfc + '\n')
    fh.close()
    util.log_write(model.logfh, 'removed alignments [%d]' %bad_count)

    ## Clean up
    os.system('rm -f %s/mfc.list.* %s/align.output.*' %(output_dir, output_dir))
    return output_dir

def map_tri_to_mono(model, root_dir, tri_mlf, mono_mlf):
    """
    Convert a triphone mlf to monophones to remove artifacts from state tying
    """
    
    cmd = 'HLEd -b -m -i %s /dev/null %s' %(mono_mlf, tri_mlf)

    if model.local == 1: os.system(cmd)
    else: util.run(cmd, '%s' %root_dir)

    return mono_mlf

def mono_to_tri(model, root_dir, mono_dir, phone_mlf, tri_mlf, mono_list, tri_list):
    """
    Convert a monophone model and phone mlf to triphones
    """

    ## Create the xword directory and the current output directory
    output_dir = '%s/HMM-0-0' %root_dir
    util.create_new_dir(root_dir)
    util.create_new_dir(output_dir)

    mktri_led = '%s/mktri_cross.led' %output_dir
    mktri_hed = '%s/mktri.hed' %output_dir
    hled_log = '%s/hled_make_tri.log' %output_dir
    hhed_log = '%s/hhed_clone_mono.log' %output_dir

    ## Create an HLEd script
    fh = open(mktri_led, 'w')
    fh.write('NB sp\n')
    fh.write('TC\n')
    fh.write('IT\n')
    fh.write('CH sil * sil *\n')
    fh.write('CH sp  * sp  *\n')
    fh.write('ME sil sil sil sil\n')
    fh.write('ME sil sil sil\n')
    fh.write('ME sil sp sil\n')
    fh.close()

    ## Create a new alignment in tri_mlf and output used triphones to tri_list
    cmd  = 'HLEd -A -n %s' %tri_list
    cmd += ' -i %s' %tri_mlf
    cmd += ' %s %s > %s' %(mktri_led, phone_mlf, hled_log)

    if model.local: os.system(cmd)
    else: util.run(cmd, output_dir)

    ## Create an HHEd script to clone monophones to triphones
    fh = open(mktri_hed, 'w')
    for line in open(mono_list):
        mono = line.strip()
        fh.write('TI T_%s {(%s).transP}\n' %(mono, mono))
    fh.write('CL %s\n' %tri_list)
    fh.close()

    ## Run HHEd to clone monophones and tie transition matricies
    cmd  = 'HHEd -A -T 1 -H %s/MMF' %mono_dir
    cmd += ' -M %s' %output_dir
    cmd += ' %s %s > %s' %(mktri_hed, mono_list, hhed_log)

    if model.local: os.system(cmd)
    else: util.run(cmd, output_dir)

    return output_dir

def init_tri_from_mono(model, root_dir, mono_dir, tri_mlf, mono_list, tri_list):
    """
    Convert a monophone model and triphone mlf to triphones
    """

    ## Create the xword directory and the current output directory
    output_dir = '%s/HMM-0-0' %root_dir
    util.create_new_dir(root_dir)
    util.create_new_dir(output_dir)

    mktri_hed = '%s/mktri.hed' %output_dir
    hhed_log = '%s/hhed_clone_mono.log' %output_dir

    ## Create an HHEd script to clone monophones to triphones
    fh = open(mktri_hed, 'w')
    for line in open(mono_list):
        mono = line.strip()
        fh.write('TI T_%s {(%s).transP}\n' %(mono, mono))
    fh.write('CL %s\n' %tri_list)
    fh.close()

    ## Run HHEd to clone monophones and tie transition matricies
    cmd  = 'HHEd -A -T 1 -H %s/MMF' %mono_dir
    cmd += ' -M %s' %output_dir
    cmd += ' %s %s > %s' %(mktri_hed, mono_list, hhed_log)

    if model.local: os.system(cmd)
    else: util.run(cmd, output_dir)

    return output_dir

def tie_states(model, output_dir, model_dir, mono_list, tri_list, tied_list):
    """
    Tie HMM states using decision tree clustering
    """

    util.create_new_dir(output_dir)
    tree_hed = '%s/tree.hed' %output_dir
    all_tri_list = '%s/all_tri.list' %model.exp
    tree_output = '%s/trees' %output_dir
    hhed_log = '%s/hhed_cluster.log' %output_dir

    ## Decision tree parameters
    ro = 200
    tb = 750

    ## Create the full list of possible triphones
    phones = open(mono_list).read().splitlines()
    non_sp_phones = [p for p in phones if p not in ['sp', 'sil']]

    fh = open(all_tri_list, 'w')
    fh.write('sp\n')
    fh.write('sil\n')
    for p1 in non_sp_phones:
        fh.write('sil-%s+sil\n' %p1)
        for p2 in non_sp_phones:
            fh.write('sil-%s+%s\n' %(p1, p2))
            fh.write('%s-%s+sil\n' %(p2, p1))
            for p3 in non_sp_phones:
                fh.write('%s-%s+%s\n' %(p2, p1, p3))
    fh.close()

    ## Set up decision tree clustering
    fh = open(tree_hed, 'w')
    fh.write('RO %d %s/stats\n' %(ro, model_dir))
    fh.write('TR 0\n')
    fh.write('%s\n' %open(model.tree_questions).read())
    fh.write('TR 12\n')
    for p in non_sp_phones:
        for s in range(1, model.states+1)[1:-1]:
            fh.write('TB %d "ST_%s_%d_" {(%s,*-%s+*,%s+*,*-%s).state[%d]}\n' %(tb,p,s,p,p,p,p,s))
    fh.write('TR 1\n')
    fh.write('AU "%s"\n' %all_tri_list)
    fh.write('CO "%s"\n' %tied_list)
    fh.write('ST "%s"\n' %tree_output)
    fh.close()

    ## Use HHEd to cluster
    cmd  = 'HHEd -A -T 1 -H %s/MMF' %model_dir
    cmd += ' -M %s' %output_dir
    cmd += ' %s %s > %s' %(tree_hed, tri_list, hhed_log)

    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)

    return output_dir

def tie_states_search(model, output_dir, model_dir, mono_list, tri_list, tied_list):
    """
    Tie HMM states using decision tree clustering
    """

    util.create_new_dir(output_dir)
    tree_hed = '%s/tree.hed' %output_dir
    tree_output = '%s/trees' %output_dir
    hhed_log = '%s/hhed_cluster.log' %output_dir
    all_tri_list = '%s/all_tri.list' %model.exp

    ## Decision tree parameters
    ro = model.dt_ro
    tb = model.dt_tb
    tb_min = 100.0
    tb_max = 10000.0

    ## Create the full list of possible triphones
    phones = open(mono_list).read().splitlines()
    non_sp_phones = [p for p in phones if p not in ['sp', 'sil']]
    fh = open(all_tri_list, 'w')
    fh.write('sp\n')
    fh.write('sil\n')
    for p1 in non_sp_phones:
        fh.write('sil-%s+sil\n' %p1)
        for p2 in non_sp_phones:
            fh.write('sil-%s+%s\n' %(p1, p2))
            fh.write('%s-%s+sil\n' %(p2, p1))
            for p3 in non_sp_phones:
                fh.write('%s-%s+%s\n' %(p2, p1, p3))
    fh.close()

    ## Search over tb arguments to get the right number states
    num_states = 0
    attempts = 0
    prev_tb = 0
    while True:

        os.system('rm -f %s %s %s' %(tree_hed, tree_output, hhed_log))
        
        ## Set up decision tree clustering
        fh = open(tree_hed, 'w')
        fh.write('RO %d %s/stats\n' %(ro, model_dir))
        fh.write('TR 0\n')
        fh.write('%s\n' %open(model.tree_questions).read())
        fh.write('TR 12\n')
        for p in non_sp_phones:
            for s in range(1, model.states+1)[1:-1]:
                fh.write('TB %d "ST_%s_%d_" {(%s,*-%s+*,%s+*,*-%s).state[%d]}\n' %(tb,p,s,p,p,p,p,s))
        fh.write('TR 1\n')
        fh.write('AU "%s"\n' %all_tri_list)
        fh.write('CO "%s"\n' %tied_list)
        fh.write('ST "%s"\n' %tree_output)
        fh.close()

        ## Use HHEd to cluster
        cmd  = 'HHEd -A -T 1 -H %s/MMF' %model_dir
        cmd += ' -M %s' %output_dir
        cmd += ' %s %s > %s' %(tree_hed, tri_list, hhed_log)

        if model.local == 1: os.system(cmd)
        else: util.run(cmd, output_dir)
        num_states = int(os.popen('grep -c "<MEAN>" %s/MMF' %output_dir).read().strip())

        
        if abs(float(num_states - model.triphone_states)/model.triphone_states) <= 0.01:
            util.log_write(model.logfh, ' current states [%d] tb [%1.2f]' %(num_states, tb))
            break
        
        if abs(prev_tb - tb) <= 0.01:
            util.log_write(model.logfh, ' Could not converge. Stopping. Current states [%d] tb [%1.2f]' %(num_states,tb))
            break
        
        attempts += 1
        prev_tb = tb
        if num_states < model.triphone_states:
            tb = (tb_min + tb) / 2
            tb_max = prev_tb
        else:
            tb = (tb_max + tb) / 2
            tb_min = prev_tb
        util.log_write(model.logfh, ' [%d] goal [%d] current states [%d] tb [%1.2f] -> [%1.2f] [%1.1f %1.1f]' %(attempts, model.triphone_states, num_states, prev_tb, tb, tb_min, tb_max))

        if attempts > 50:
            util.log_write(model.logfh, ' Goal not reached after 50 tries. Exiting.')
            sys.exit()

    return output_dir

def diagonalize(model, output_dir, model_dir, model_list, mlf_file, mix_size):
    """
    Diagonalize output distributions
    """
    util.create_new_dir(output_dir)

    diag_config = '%s/config.diag' %output_dir
    global_class = '%s/global' %output_dir

    fh = open(diag_config, 'w')
    fh.write('HADAPT:TRANSKIND = SEMIT\n')
    fh.write('HADAPT:USEBIAS = FALSE\n')
    fh.write('HADAPT:BASECLASS = global\n')
    fh.write('HADAPT:SPLITTHRESH = 0.0\n')
    fh.write('HADAPT:MAXXFORMITER = 100\n')
    fh.write('HADAPT:MAXSEMITIEDITER = 20\n')
    fh.write('HADAPT:TRACE = 61\n')
    fh.write('HMODEL:TRACE = 512\n')
    fh.write('HADAPT: SEMITIED2INPUTXFORM = TRUE\n')
    fh.close()

    max_mix = 2 * mix_size
    fh = open(global_class, 'w')
    fh.write('~b "global"\n')
    fh.write('<MMFIDMASK> *\n')
    fh.write('<PARAMETERS> MIXBASE\n')
    fh.write('<NUMCLASSES> 1\n')
    fh.write('<CLASS> 1 {*.state[2-4].mix[1-%d]}\n' %max_mix)
    fh.close()

    extra = ' -C %s -J %s -K %s/HMM-%d-0 -u stw' %(diag_config, output_dir, output_dir, mix_size)

    hmm_dir, k, likelihood = run_iter(model, output_dir, model_dir, mlf_file, model_list, mix_size, 0, extra)

    return hmm_dir, likelihood

def make_hvite_xword_config(model, config_file, target_kind):
    """
    Make a xword config file for hvite
    """

    fh = open(config_file, 'w')
    fh.write('HPARM: TARGETKIND = %s\n' %target_kind)
    fh.write('FORCECXTEXP = T\n')
    fh.write('ALLOWXWRDEXP = T\n')
    fh.close()

    return config_file

    
