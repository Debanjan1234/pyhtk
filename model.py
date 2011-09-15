"""
A Python package for building HMM models for speech recognition using HTK
Daniel Gillick (dgillick@gmail.com)

----------
Dependencies:
    HTK (tested with version 3.4)
    sph2pipe (for audio processing)
    Python (tested with 2.6.6)

----------
Inputs:
    Dictionary
    List of WAV files
    Transcriptions
    Decision Tree features
    Feature config file

Main input file format (setup):

<input wav file> <config file> <word transcription>

"""

import os, sys, re, random, time, gzip
import util
from util import log_write as log


class Model:
    def __init__(self, config, train=False):
        """
        Initialize an HTK object from a parsed config file

        exp               [experiment directory path]
        data              [data directory path]
        dict              [dictionary path]
        tree_questions    [tree questions path]
        setup             [setup file path]
        local             [only run locally (ignore jobs)]
        jobs              [max number of parallel jobs to create]
        verbose           [0, 1, 2, ...]
        """

        self.config = config

        ## Load training pipeline
        self.train_pipeline = {}
        if train:
            self.train_pipeline['clean'] = int(config.get('train_pipeline', 'clean'))
            self.train_pipeline['coding'] = int(config.get('train_pipeline', 'coding'))
            self.train_pipeline['lm'] = int(config.get('train_pipeline', 'lm'))
            self.train_pipeline['flat_start'] = int(config.get('train_pipeline', 'flat_start'))
            self.train_pipeline['mono_to_tri'] = int(config.get('train_pipeline', 'mono_to_tri'))
            self.train_pipeline['mixup_tri'] = int(config.get('train_pipeline', 'mixup_tri'))
            self.train_pipeline['mmi'] = int(config.get('train_pipeline', 'mmi'))

        ## Create experiment directory and a new log file
        self.exp = config.get('paths', 'exp')
        if self.train_pipeline and self.train_pipeline['clean']: os.system('rm -rf %s' %self.exp)
        if not os.path.isdir(self.exp): os.makedirs(self.exp)
        self.log = '%s/log' %self.exp
        if os.path.isfile(self.log):
            self.logfh = open(self.log, 'a')
        else:
            self.logfh = open(self.log, 'w')

        self.data = config.get('paths', 'data')
        if not os.path.isdir(self.data): os.makedirs(self.data)
        
        ## Load the other paths
        self.common = config.get('paths', 'common')
        self.orig_dict = config.get('paths', 'dict')
        self.tree_questions = config.get('paths', 'tree_questions')
        self.mfc_config = config.get('paths', 'mfc_config')
        self.setup = config.get('paths', 'setup')
        if not self.setup.endswith('gz'): self.setup_length = int(os.popen('wc -l %s' %self.setup).read().split()[0])
        else: self.setup_length = int(os.popen('zcat %s | wc -l' %self.setup).read().split()[0])

        ## Load settings
        self.local = int(config.get('settings', 'local'))
        self.jobs = int(config.get('settings', 'jobs'))
        self.verbose = int(config.get('settings', 'verbose'))

        ## Load HMM parameters
        self.states = int(config.get('hmm_params', 'states'))
        self.triphone_states = int(config.get('hmm_params', 'triphone_states'))
        self.dt_ro = float(config.get('hmm_params', 'dt_ro'))
        self.dt_tb = float(config.get('hmm_params', 'dt_tb'))

        ## Load training parameters
        self.split_path_letters = int(config.get('train_params', 'split_path_letters'))
        self.initial_mono_iters = int(config.get('train_params', 'initial_mono_iters'))
        self.mono_iters = int(config.get('train_params', 'mono_iters'))
        self.initial_tri_iters = int(config.get('train_params', 'initial_tri_iters'))
        self.tri_iters = int(config.get('train_params', 'tri_iters'))
        self.tri_mixup_schedule = map(int, config.get('train_params', 'tri_mixup_schedule').split('_'))
        self.tri_iters_per_split = int(config.get('train_params', 'tri_iters_per_split'))

        ## Shared files created during training
        self.mfc_list = '%s/mfc.list' %self.exp
        self.mono_root = '%s/Mono' %self.exp
        self.xword_root = '%s/Xword' %self.exp
        self.train_dict = '%s/train_dict' %self.exp
        self.decode_dict = '%s/decode_dict' %self.exp
        self.lm_dir = '%s/LM' %self.exp
        self.lm = '%s/lm' %self.exp
        self.lm_order = 3
        self.mmi_lm = '%s/mmi_lm' %self.exp
        self.proto_hmm = '%s/proto_hmm' %self.exp
        self.word_mlf = '%s/words.mlf' %self.exp
        self.phone_mlf = '%s/phone.mlf' %self.exp
        self.tri_mlf = '%s/tri.mlf' %self.exp
        self.phone_list = '%s/mono.list' %self.exp
        self.tri_list = '%s/tri.list' %self.exp
        self.tied_list = '%s/tied.list' %self.exp

    def train(self):

        ## Copy config file to the experiment dir
        config_output = '%s/config' %self.exp
        self.config.write(open(config_output, 'w'))
        log(self.logfh, 'TRAINING with config [%s]' %config_output)

        if self.train_pipeline['coding']:
            log(self.logfh, 'CODING started')
            import coding
            coding_dir = '%s/Coding' %self.exp
            util.create_new_dir(coding_dir)
            count = coding.wav_to_mfc(model, coding_dir, self.mfc_list)
            log(self.logfh, 'CODING finished [%d files]' %count)

        if self.train_pipeline['lm']:
            log(self.logfh, 'MLF/LM/DICT started')
            import dict_and_lm
            num_utts, words = dict_and_lm.make_mlf_from_transcripts(self, self.orig_dict, self.setup, self.data, self.word_mlf, self.mfc_list)
            log(self.logfh, '  wrote word mlf [%d utts] [%s]' %(num_utts, self.word_mlf))

            num_entries = dict_and_lm.make_train_dict(self.orig_dict, self.train_dict, words)
            dict_and_lm.make_decode_dict(self.orig_dict, self.decode_dict, words)
            log(self.logfh, '  wrote training dictionary [%d entries] [%s]' %(num_entries, self.train_dict))

            util.create_new_dir(self.lm_dir)
            train_vocab = '%s/vocab' %self.lm_dir
            ppl = dict_and_lm.build_lm_from_mlf(self, self.word_mlf, self.train_dict, train_vocab, self.lm_dir, self.lm, self.lm_order)
            log(self.logfh, '  wrote lm [%s] training ppl [%1.2f]' %(self.lm, ppl))
            log(self.logfh, 'MLF/LM/DICT finished')
            
        if self.train_pipeline['flat_start']:
            import init_hmm
            init_hmm.word_to_phone_mlf(self, self.train_dict, self.word_mlf, self.phone_mlf, self.phone_list)
            log(self.logfh, 'wrote phone mlf [%s]' %self.phone_mlf)

            init_hmm.make_proto_hmm(self, self.mfc_list, self.proto_hmm)
            hmm_dir = init_hmm.initialize_hmms(self, self.mono_root, self.mfc_list, self.phone_list, self.proto_hmm)
            log(self.logfh, 'initialized an HMM for each phone in [%s]' %hmm_dir)

            import train_hmm
            for iter in range(1, self.initial_mono_iters+1):
                hmm_dir, k, L = train_hmm.run_iter(self, self.mono_root, hmm_dir, self.phone_mlf, self.phone_list, 1, iter)
                log(self.logfh, 'ran an iteration of BW in [%s] lik/fr [%1.4f]' %(hmm_dir, L))
            
            align_dir = train_hmm.align(self, self.mono_root, self.mfc_list, hmm_dir, self.word_mlf, self.phone_mlf, self.phone_list, self.train_dict)
            log(self.logfh, 'aligned with model in [%s], wrote phone mlf [%s]' %(hmm_dir, self.phone_mlf))

            for iter in range(self.initial_mono_iters+1, self.initial_mono_iters+1+self.mono_iters):
                hmm_dir, k, L = train_hmm.run_iter(self, self.mono_root, hmm_dir, self.phone_mlf, self.phone_list, 1, iter)
                log(self.logfh, 'ran an iteration of BW in [%s] lik/fr [%1.4f]' %(hmm_dir, L))

            log(self.logfh, 'FLAT_START finished')

        if self.train_pipeline['mono_to_tri']:
            import train_hmm
            mono_final_dir = '%s/HMM-%d-%d' %(self.mono_root, 1, self.initial_mono_iters+self.mono_iters)
            hmm_dir = train_hmm.mono_to_tri(self, self.xword_root, mono_final_dir, self.phone_mlf, self.tri_mlf, self.phone_list, self.tri_list)
            log(self.logfh, 'initialized triphone models in [%s], created triphone mlf [%s]' %(hmm_dir, self.tri_mlf))
            
            for iter in range(1, self.initial_tri_iters+1):
                hmm_dir, k, L = train_hmm.run_iter(self, self.xword_root, hmm_dir, self.tri_mlf, self.tri_list, 1, iter)
                log(self.logfh, 'ran an iteration of BW in [%s] lik/fr [%1.4f]' %(hmm_dir, L))
            
            xword_tie_dir = '%s/HMM-%d-%d' %(self.xword_root, 1, self.initial_tri_iters+1)
            hmm_dir = train_hmm.tie_states_search(self, xword_tie_dir, hmm_dir, self.phone_list, self.tri_list, self.tied_list)
            log(self.logfh, 'tied states in [%s]' %hmm_dir)

            hmm_dir = '%s/HMM-%d-%d' %(self.xword_root, 1, 2)
            for iter in range(self.initial_tri_iters+2, self.initial_tri_iters+1+self.tri_iters+1):
                hmm_dir, k, L = train_hmm.run_iter(self, self.xword_root, hmm_dir, self.tri_mlf, self.tied_list, 1, iter)
                log(self.logfh, 'ran an iteration of BW in [%s] lik/fr [%1.4f]' %(hmm_dir, L))

            log(self.logfh, 'MONO_TO_TRI finished')

        if self.train_pipeline['mixup_tri']:
            import train_hmm

            ## mixup everything
            start_gaussians = 1
            start_iter = self.initial_tri_iters+self.tri_iters+1
            hmm_dir = '%s/HMM-%d-%d' %(self.xword_root, start_gaussians, start_iter)
            for mix_size in self.tri_mixup_schedule:
                hmm_dir = train_hmm.mixup(self, self.xword_root, hmm_dir, self.tied_list, mix_size)
                log(self.logfh, 'mixed up to [%d] in [%s]' %(mix_size, hmm_dir))
                for iter in range(1, self.tri_iters_per_split+1):
                    hmm_dir, k, L = train_hmm.run_iter(self, self.xword_root, hmm_dir, self.tri_mlf, self.tied_list, mix_size, iter)
                    log(self.logfh, 'ran an iteration of BW in [%s] lik/fr [%1.4f]' %(hmm_dir, L))

        if self.train_pipeline['mmi']:

            ## Common items
            import mmi
            mmi_dir = '%s/MMI' %self.exp
            util.create_new_dir(mmi_dir)
            mfc_list_mmi = '%s/mfc.list' %mmi_dir
            os.system('cp %s %s' %(self.mfc_list, mfc_list_mmi))
            hdecode_config = '%s/config.hdecode' %self.common
            hlrescore_config = '%s/config.hlrescore' %self.common
            hmmirest_config = '%s/config.hmmirest' %self.common

            ## Create weak LM
            import dict_and_lm
            train_vocab = '%s/vocab' %self.lm_dir
            lm_order = 2
            target_ppl_ratio = 8
            ppl = dict_and_lm.build_lm_from_mlf(self, self.word_mlf, self.train_dict, train_vocab, self.lm_dir, self.mmi_lm, lm_order, target_ppl_ratio)
            log(self.logfh, 'wrote lm for mmi [%s] training ppl [%1.2f]' %(self.mmi_lm, ppl))

            ## Create decoding lattices for every utterance
            lattice_dir = '%s/Denom/Lat_word' %mmi_dir
            util.create_new_dir(lattice_dir)
            num_gaussians = self.tri_mixup_schedule[-1]
            iter_num = self.tri_iters_per_split
            model_dir = '%s/HMM-%d-%d' %(self.xword_root, num_gaussians, iter_num)
            mmi.decode_to_lattices(model, lattice_dir, model_dir, mfc_list_mmi, self.mmi_lm, self.decode_dict,
                                   self.tied_list, hdecode_config, self.word_mlf)
            log(self.logfh, 'generated training lattices in [%s]' %lattice_dir)

            ## Prune and determinize lattices
            pruned_lattice_dir = '%s/Denom/Lat_prune' %mmi_dir
            util.create_new_dir(pruned_lattice_dir)
            mmi.prune_lattices(model, lattice_dir, pruned_lattice_dir, self.decode_dict, hlrescore_config)
            log(self.logfh, 'pruned lattices in [%s]' %pruned_lattice_dir)

            ## Phone-mark lattices
            phone_lattice_dir = '%s/Denom/Lat_phone' %mmi_dir
            util.create_new_dir(phone_lattice_dir)
            mmi.phonemark_lattices(model, pruned_lattice_dir, phone_lattice_dir, model_dir, mfc_list_mmi,
                                   self.mmi_lm, self.decode_dict, self.tied_list, hdecode_config)
            log(self.logfh, 'phone-marked lattices in [%s]' %phone_lattice_dir)

            ## Create numerator word lattices
            num_lattice_dir = '%s/Num/Lat_word' %mmi_dir
            util.create_new_dir(num_lattice_dir)
            mmi.create_num_lattices(model, num_lattice_dir, self.mmi_lm, self.decode_dict, hlrescore_config, self.word_mlf)
            log(self.logfh, 'generated numerator lattices in [%s]' %num_lattice_dir)

            ## Phone-mark numerator lattices
            num_phone_lattice_dir = '%s/Num/Lat_phone' %mmi_dir
            util.create_new_dir(num_phone_lattice_dir)
            mmi.phonemark_lattices(model, num_lattice_dir, num_phone_lattice_dir, model_dir, mfc_list_mmi,
                                   self.mmi_lm, self.decode_dict, self.tied_list, hdecode_config)
            log(self.logfh, 'phone-marked numerator lattices in [%s]' %num_phone_lattice_dir)

            ## Add LM scores to numerator phone lattices
            num_phone_lm_lattice_dir = '%s/Num/Lat_phone_lm' %mmi_dir
            util.create_new_dir(num_phone_lm_lattice_dir)
            mmi.add_lm_lattices(model, num_phone_lattice_dir, num_phone_lm_lattice_dir, self.decode_dict, self.mmi_lm, hlrescore_config)
            log(self.logfh, 'added LM scores to numerator lattices in [%s]' %num_phone_lm_lattice_dir)

            ## Modified Baum-Welch estimation
            root_dir = '%s/Models' %mmi_dir
            util.create_new_dir(root_dir)
            mmi_iters = 12
            mix_size = num_gaussians
            for iter in range(1, mmi_iters+1):
                model_dir = mmi.run_iter(model, model_dir, num_phone_lm_lattice_dir, phone_lattice_dir, root_dir,
                                         self.tied_list, mfc_list_mmi, hmmirest_config, mix_size, iter)
                log(self.logfh, 'Ran an iteration of Modified BW in [%s]' %model_dir)
                        
if __name__ == '__main__':

    from optparse import OptionParser
    usage = 'Usage: Python %s [options] <config>' %sys.argv[0]
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()

    if len(args) < 1:
        sys.stderr.write('%s\n' %usage)
        sys.exit()
        
    import ConfigParser
    config = ConfigParser.ConfigParser()
    config.read(args[0])

    ## Training
    model = Model(config, options)
    start_time = time.time()
    model.train()
    total_time = time.time() - start_time
    print 'time elapsed [%1.2f]' %total_time
    
