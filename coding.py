"""
Coding functions
"""

import os, sys, gzip
import util

def create_config(model):
    """
    Create an HTK style config from the front end settings
    """
    targetkind = 'MFCC'
    if model.use_c0: targetkind += '_0'
    if model.use_deltas: targetkind += '_D'
    if model.use_ddeltas: targetkind += '_A'
    if model.mean_norm: targetkind += '_Z'

    model.front_end = {'TARGETKIND': targetkind,
                       'TARGETRATE': '%1.1f' %(model.frame_length * 10000.0),
                       'SAVECOMPRESSED': 'T',
                       'SAVEWITHCRC': 'T',
                       'WINDOWSIZE': '%1.1f' %(model.delta_window * 10000.0),
                       'USEHAMMING': 'T',
                       'PREEMCOEF': '0.97',
                       'NUMCHANS': '26',
                       'CEPLIFTER': '22',
                       'NUMCEPS': '%d' %model.num_cepstra,
                       'ENORMALISE': 'T',
                       'ZMEANSOURCE': 'T',
                       'USEPOWER': 'T',
                       'BYTEORDER': 'VAX'}

    fh = open(model.mfc_config, 'w')
    for key, val in model.front_end.items():
        fh.write('%s = %s\n' %(key, val))
    fh.close()


def get_mfc_name_from_wav(wav, path, just_key=False):
    items = wav.split('/')
    basename = items.pop()
    new_path = basename
    count = 0
    while True:
        count += 1
        dir = items.pop()
        if count < 2 or dir in basename or dir.replace('_', '') in basename: new_path = dir + '/' + new_path
        else: break

    if just_key: return new_path

    new_path = '%s/%s.mfc' %(path, new_path.split('.')[0])
    dir = os.path.dirname(new_path)
    if not os.path.isdir(dir): os.makedirs(dir)
    return new_path


def wav_to_mfc(model, output_dir, mfc_list):
    """
    Use HCopy to code each wav in the setup file. HCopy takes a config
    file (-C) and an input (-S). The input is a file where each line
    looks like:
    <wav file> <mfc file>
    """

    def hcopy(config, input):
        cmd = 'HCopy -A -T 1 -C %s -C %s -S %s' %(model.mfc_config, config, input)
        return cmd

    ## Create list files for HCopy <wav file> <mfc file>
    lines_per_split = 500
    count = 0
    prev_config = ''
    cmds = []
    mfcs = []
    file = '%s/hcopy.list.0' %output_dir
    fh = open(file, 'w')

    if model.setup.endswith('gz'): setup_reader = lambda x: gzip.open(x)
    else: setup_reader = lambda x: open(x)
    
    for line in setup_reader(model.setup):
        count += 1
        [wav, config] = line.strip().split()[0:2]
        if not os.path.isfile(wav): sys.stderr.write('missing [%s]\n' %wav)
        mfc = get_mfc_name_from_wav(wav, model.data)
        mfcs.append(mfc)

        if count > 1 and (count % lines_per_split == 0 or config != prev_config):
            cmds.append(hcopy(prev_config, file))
            fh.close()
            file = '%s/hcopy.list.%d' %(output_dir, len(cmds))
            fh = open(file, 'w')

        fh.write('%s %s\n' %(wav, mfc))
        prev_config = config

    cmds.append(hcopy(config, file))
    fh.close()

    ## Non-parallel case
    if model.local == 1:
        for cmd in cmds: os.system(cmd)

    ## Parallel case: one command per line in cmds_file
    else:
        cmds_file = '%s/hcopy.commands' %output_dir
        fh = open(cmds_file, 'w')
        for cmd in cmds: fh.write('%s\n' %cmd)
        fh.close()
        util.run_parallel(cmds_file, model.jobs, output_dir)

    ## Create a file listing all created MFCs
    fh = open(mfc_list, 'w')
    for mfc in mfcs:
        fh.write('%s\n' %mfc)
    fh.close()

    ## Clean up
    os.system('rm -f %s/hcopy.list.*' %output_dir)
    return count
