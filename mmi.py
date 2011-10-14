"""
Functions for running MMI
"""

import os, sys
import util

class SplitList:

    def __init__(self, output_dir, file_list, by_path=True, by_letters=0):
        self.file_list = []
        self.key_by_split_file = {}
        
        fh = {}
        file_data = {}
        index = 0
        for line in open(file_list):
            file = line.strip()
            items = file.split('/')

            ## Get the key to use for splitting
            if by_path: split_key = items[-2]
            elif by_letters: 
                split_key = items[-1][:by_letters]
            if split_key not in fh:
                index += 1
                split_file = '%s/list.%d' %(output_dir, index)
                self.file_list.append(split_file)
                fh[split_key] = split_file
                self.key_by_split_file[split_file] = split_key

            if fh[split_key] not in file_data: file_data[fh[split_key]] = []
            file_data[fh[split_key]].append(file)
                
        for key, files in file_data.items():
            handle = open(key, 'w')
            for file in files: handle.write('%s\n' %file)
            handle.close()

    def get_files(self):
        return self.file_list

    def get_key(self, file):
        return self.key_by_split_file[file]
    
    def cleanup(self):
        for file in self.file_list: os.remove(file)

def decode_to_lattices(model, output_dir, model_dir, mfc_list, lm, dict, model_list, gold_mlf):

    sys.stderr.write('Decoding to lattices\n')
    output_mlf = '%s/train_recog.mlf' %output_dir
    results_log = '%s/results.log' %output_dir

    ## Create a config file to use with HDecode
    hdecode_config = '%s/hdecode.config' %output_dir
    fh = open(hdecode_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HPARM: TARGETKIND = MFCC_0_D_A_Z\n')
    fh.write('GCFREQ = 50\n')
    fh.write('HLAT:TRACE = 19\n')
    fh.write('HLVNET:TRACE = 1\n')
    fh.write('HLVREC:TRACE = 1\n')
    fh.write('HLVLM:TRACE = 1\n')
    fh.write('LATPRUNEBEAM = 500.0\n')
    fh.write('MAXLMLA = 3.0\n')
    fh.write('BUILDLATSENTEND = T\n')
    fh.write('FORCELATOUT = F\n')
    fh.write('STARTWORD = <s>\n')
    fh.write('ENDWORD = </s>\n')
    fh.close()

    ## HDecode parameters
    utts_per_split = 100
    block_size = 5
    beam = 150.0
    word_end_beam = 125.0
    max_model = 10000
    lm_scale = 15.0
    word_insertion_penalty = 0.0

    def hdecode(input, output):
        cmd  = 'HDecode -A -D -V -T 9 -o M -z lat -C %s' %hdecode_config
        cmd += ' -H %s/MMF' %model_dir
        cmd += ' -k %d' %block_size
        cmd += ' -t %f 100.0' %beam
        cmd += ' -v %f 115.0' %word_end_beam
        cmd += ' -u %d' %max_model
        cmd += ' -s %f' %lm_scale
        cmd += ' -p %f' %word_insertion_penalty
        cmd += ' -w %s' %lm
        cmd += ' -S %s' %input
        cmd += ' -l %s/' %output
        cmd += ' %s %s' %(dict, model_list)
        if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
        return cmd
    
    ## Split up MFC list
    split_mfc = SplitList(output_dir, mfc_list, by_path=True)

    ## Create the HDecode commands
    cmds = []
    inputs = split_mfc.get_files()
    for input in inputs:
        output = '%s/%s' %(output_dir, split_mfc.get_key(input))
        if not os.path.isdir(output): os.makedirs(output)
        cmds.append(hdecode(input, output))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hdecode.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

    ## Copy old mfc list
    old_mfc_list = '%s/mfc_old.list' %output_dir
    os.system('cp %s %s' %(mfc_list, old_mfc_list))

    ## Prune bad lats from the mfc list
    lat_ids = [os.path.basename(f).split('.')[0] for f in util.get_files(output_dir, r'.*\.lat')]
    bad_count = 0
    fh = open(mfc_list, 'w')
    for mfc in open(old_mfc_list):
        id = os.path.basename(mfc.strip()).split('.')[0]

        ## Check for missing transcriptions
        if id not in lat_ids:
            if model.verbose > 1: util.log_write(model.logfh, 'removed bad lat [%s]' %id)
            bad_count += 1
        else: fh.write(mfc)
    fh.close()
    util.log_write(model.logfh, 'removed bad lats [%d]' %bad_count)
    
    ## Create an MLF from the recognition output
    outputs = util.get_files(output_dir, r'.*\.rec')
    os.popen('rm -f %s' %output_mlf)
    fh = open(output_mlf, 'w')
    fh.write('#!MLF!#\n')
    for output in outputs:
        fh.write('"%s"\n' %output)
        for line in open(output):
            if '<s>' in line or '</s>' in line: continue
            fh.write(line)
        fh.write('.\n')
    fh.close()

    ## Evaluate
    cmd  = 'HResults -h -n -A -T 1'
    cmd += ' -I %s' %gold_mlf
    cmd += ' %s %s > %s' %(model_list, output_mlf, results_log)
    os.system(cmd)
    print os.popen('cat ' + results_log).read()


def prune_lattices(model, lattice_dir, output_dir, dict):

    sys.stderr.write('Pruning lattices\n')
    
    ## Create a config file to use with HLRescore
    hlrescore_config = '%s/hlrescore.config' %output_dir
    fh = open(hlrescore_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HLRESCORE: FIXBADLATS = TRUE\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.close()

    ## HLRescore parameters
    utts_per_split = 100
    pruning_threshold = 200.0
    grammar_scale = 15.0
    trans_penalty = 0.0

    def hlrescore(input, path):
        cmd  = 'HLRescore -A -D -T 1 -w -m f -C %s' %hlrescore_config
        cmd += ' -S %s' %input
        cmd += ' -t %f 200.0' %pruning_threshold
        cmd += ' -L %s/%s/' %(lattice_dir, path)
        cmd += ' -l %s/%s/' %(output_dir, path)
        cmd += ' -s %f' %grammar_scale
        cmd += ' -p %f' %trans_penalty
        cmd += ' %s' %dict
        if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
        return cmd

    ## Split up lattice list
    lattice_list = '%s/lattice.list' %output_dir
    fh = open(lattice_list, 'w')
    remove_gz = lambda x: x.replace('.gz', '')
    files = map(remove_gz, util.get_files(lattice_dir, r'.*\.lat'))
    fh.write('\n'.join(files))
    fh.close()
    split_lattice = SplitList(output_dir, lattice_list, by_path=True)

    ## Create the HLRescore commands
    cmds = []
    inputs = split_lattice.get_files()
    for input in inputs:
        key = split_lattice.get_key(input)
        new_output = '%s/%s' %(output_dir, key)
        if not os.path.isdir(new_output): os.makedirs(new_output)
        cmds.append(hlrescore(input, key))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hlrescore.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)


def phonemark_lattices(model, lattice_dir, output_dir, model_dir, mfc_list, lm, dict, model_list):

    sys.stderr.write('Phonemarking lattices\n')

    ## Create a config file to use with HDecode
    hdecode_config = '%s/hdecode.config' %output_dir
    fh = open(hdecode_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HPARM: TARGETKIND = MFCC_0_D_A_Z\n')
    fh.write('GCFREQ = 50\n')
    fh.write('HLAT:TRACE = 19\n')
    fh.write('HLVNET:TRACE = 1\n')
    fh.write('HLVREC:TRACE = 1\n')
    fh.write('HLVLM:TRACE = 1\n')
    fh.write('LATPRUNEBEAM = 500.0\n')
    fh.write('MAXLMLA = 3.0\n')
    fh.write('BUILDLATSENTEND = T\n')
    fh.write('FORCELATOUT = F\n')
    fh.write('STARTWORD = <s>\n')
    fh.write('ENDWORD = </s>\n')
    fh.close()
    
    ## HDecode parameters
    utts_per_split = 100
    block_size = 5
    beam = 200.0
    lm_scale = 15.0
    word_insertion_penalty = 0.0

    def hdecode_mod(input, path):
        input_dir = '%s/%s/' %(lattice_dir, path)
        if not os.path.isdir(input_dir):
            input_dir = '%s/%s/' %(lattice_dir, path.replace('_', ''))
        cmd  = 'HDecode.mod -A -D -V -T 9 -q tvaldm -z lat -X lat -C %s' %hdecode_config
        cmd += ' -H %s/MMF' %model_dir
        cmd += ' -k %d' %block_size
        cmd += ' -t %f' %beam
        cmd += ' -s %f' %lm_scale
        cmd += ' -p %f' %word_insertion_penalty
        cmd += ' -w' # %s' %lm
        cmd += ' -S %s' %input
        cmd += ' -l %s/%s/' %(output_dir, path)
        cmd += ' -L %s' %input_dir
        cmd += ' %s %s' %(dict, model_list)
        if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
        return cmd

    ## Split up MFC list with unix split
    split_mfc = SplitList(output_dir, mfc_list, by_path=True)

    ## Create the HDecode commands
    cmds = []
    inputs = split_mfc.get_files()
    for input in inputs:
        key = split_mfc.get_key(input)
        new_output = '%s/%s' %(output_dir, key)
        if not os.path.isdir(new_output): os.makedirs(new_output)
        
        cmds.append(hdecode_mod(input, key))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hdecode_mod.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)
        
    ## Copy old mfc list
    old_mfc_list = '%s/mfc_old.list' %output_dir
    os.system('cp %s %s' %(mfc_list, old_mfc_list))
        
    ## Prune bad lats from the mfc list
    lat_ids = [os.path.basename(f).split('.')[0] for f in util.get_files(output_dir, r'.*\.lat')]
    bad_count = 0
    fh = open(mfc_list, 'w')
    for mfc in open(old_mfc_list):
        id = os.path.basename(mfc.strip()).split('.')[0]

        ## Check for missing transcriptions
        if id not in lat_ids:
            if model.verbose > 1: util.log_write(model.logfh, 'removed bad lat [%s]' %id)
            bad_count += 1
        else: fh.write(mfc)
    fh.close()
    util.log_write(model.logfh, 'removed bad lats [%d]' %bad_count)

def create_num_lattices(model, output_dir, lm, dict, word_mlf):

    sys.stderr.write('Creating numerator word lattices\n')

    ## Create a config file to use with HLRescore
    hlrescore_config = '%s/hlrescore.config' %output_dir
    fh = open(hlrescore_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HLRESCORE: FIXBADLATS = TRUE\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.close()
    
    def hlrescore(input, output):
        cmd  = 'HLRescore -A -D -T 1 -w -f -q tvalqr -C %s' %hlrescore_config
        cmd += ' -S %s' %input
        cmd += ' -I %s' %word_mlf
        cmd += ' -l %s/' %output
        cmd += ' %s' %dict
        if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
        return cmd

    ## Split the word mlf labels to create inputs for HLRescore
    label_list = '%s/labels.list' %output_dir
    cmd = 'grep "\.lab" %s > %s' %(word_mlf, label_list)
    os.system(cmd)
    split_label = SplitList(output_dir, label_list, by_path=False, by_letters=model.split_path_letters)

    ## Create the HLRescore commands
    cmds = []
    inputs = split_label.get_files()
    for input in inputs:
        output = '%s/%s' %(output_dir, split_label.get_key(input))
        if not os.path.isdir(output): os.makedirs(output)
        cmds.append(hlrescore(input, output))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hlrescore.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

def add_lm_lattices(model, lattice_dir, output_dir, dict, lm):

    sys.stderr.write('adding LM scores to numerator lattices\n')

    ## Create a config file to use with HLRescore
    hlrescore_config = '%s/hlrescore.config' %output_dir
    fh = open(hlrescore_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HLRESCORE: FIXBADLATS = TRUE\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: STARTWORD = <s>\n')
    fh.write('HLRESCORE: ENDWORD = </s>\n')
    fh.close()
    
    ## HLRescore parameters
    grammar_scale = 15.0
    trans_penalty = 0.0

    def hlrescore(input, path):
        cmd  = 'HLRescore -A -D -T 1 -w -c -q tvaldm -C %s' %hlrescore_config
        cmd += ' -S %s' %input
        cmd += ' -L %s/%s/' %(lattice_dir, path)
        cmd += ' -l %s/%s/' %(output_dir, path)
        cmd += ' -s %f' %grammar_scale
        cmd += ' -p %f' %trans_penalty
        cmd += ' -n %s' %lm
        cmd += ' %s' %dict
        if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
        return cmd

    ## Split up lattice list
    lattice_list = '%s/lattice.list' %output_dir
    fh = open(lattice_list, 'w')
    remove_gz = lambda x: x.replace('.gz', '')
    files = map(remove_gz, util.get_files(lattice_dir, r'.*\.lat'))
    fh.write('\n'.join(files))
    fh.close()
    split_lattice = SplitList(output_dir, lattice_list, by_path=True)

    ## Create the HLRescore commands
    cmds = []
    inputs = split_lattice.get_files()
    for input in inputs:
        key = split_lattice.get_key(input)
        new_output = '%s/%s' %(output_dir, key)
        if not os.path.isdir(new_output): os.makedirs(new_output)
        cmds.append(hlrescore(input, key))

    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd).read()
    else:
        cmds_file = '%s/hlrescore.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)


def run_iter(model, model_dir, num_lattice_dir, den_lattice_dir, root_dir, model_list, mfc_list, config, mix_size, iter):
    """
    Run an iteration of modified Baum-Welch training using HMMIRest
    """

    output_dir = '%s/HMMI-%d-%d' %(root_dir, mix_size, iter)
    util.create_new_dir(output_dir)
    utts_per_split = max(250, (1 + (model.setup_length / 200)))

    ## Create a config file to use with HLRescore
    hmmirest_config = '%s/hmmirest.config' %output_dir
    fh = open(hmmirest_config, 'w')
    #fh.write('HLANGMODFILTER = "gunzip -c $.gz"\n')
    fh.write('HNETFILTER = "gunzip -c < $.gz\n')
    fh.write('HNETOFILTER = "gzip -c > $.gz"\n')
    fh.write('RAWMITFORMAT = T\n')
    fh.write('HPARM: TARGETKIND = MFCC_0_D_A_Z\n')
    fh.write('HMMDEFOFILTER = "gzip -c > $.gz"\n')
    fh.write('HMMDEFFILTER = "gunzip -c < $.gz"\n')
    fh.write('HMMIREST: LATMASKNUM = */%%%?????.???\n')
    fh.write('HMMIREST: LATMASKDEN = */%%%?????.???\n')
    #fh.write('HMMIREST: LATMASKNUM =  */%%%%%%%%/???????????????????.???\n')
    #fh.write('HMMIREST: LATMASKDEN =  */%%%%%%%%/???????????????????.???\n')
    fh.write('HFBLAT: LATPROBSCALE = 0.06667\n')
    fh.write('HMMIREST: E = 2.0\n')
    fh.write('ISMOOTHTAU = 50\n')
    fh.write('MPE = TRUE\n')
    #fh.write('MWE = TRUE\n')
    fh.close()

    def hmmirest(input, split_num):
        cmd  = 'HMMIRest -A -D -T 1 -C %s' %hmmirest_config
        cmd += ' -H %s/MMF' %model_dir
        cmd += ' -q %s' %num_lattice_dir
        cmd += ' -r %s' %den_lattice_dir
        if split_num == 0:
            cmd += ' -u mv'
        cmd += ' -p %d' %split_num
        cmd += ' -S %s' %input
        cmd += ' -M %s %s' %(output_dir, model_list)
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
        cmds.append(hmmirest(input, split_num))

    ## Non-parallel case
    if model.local == 1:
        for cmd in cmds:
            print cmd
            print os.popen(cmd)

    ## Parallel case: one command per line in cmds_file
    else:
        cmds_file = '%s/hmmirest.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

    ## Gather the created .acc files
    acc_file = '%s/hmmirest.list' %output_dir
    os.system('ls %s/HDR*.acc* > %s' %(output_dir, acc_file))

    ## Combine acc files into a new HMM
    cmd = hmmirest(acc_file, 0)
    cmd += ' >> %s/hmmirest.log' %output_dir
    if model.local == 1: os.system(cmd)
    else: util.run(cmd, output_dir)
    
    ## Clean up
    #os.system('rm -f %s/mfc.list.* %s/HER*.acc' %(output_dir, output_dir))

    return output_dir
