import os, sys, time, re
from model import Model
import util
from util import log_write as log

class Decoder:

    def __init__(self, config, model=None):

        self.config = config
        self.setup = config.get('paths', 'setup')
        self.exp = config.get('paths', 'exp')
        self.mfc_config = config.get('paths', 'mfc_config')
        if not os.path.isdir(self.exp): os.makedirs(self.exp)
        self.data = config.get('paths', 'data')
        if not os.path.isdir(self.data): os.makedirs(self.data)

        self.log = '%s/log' %self.exp
        if os.path.isfile(self.log):
            self.logfh = open(self.log, 'a')
        else:
            self.logfh = open(self.log, 'w')

        ## Load settings
        self.local = int(config.get('settings', 'local'))
        self.jobs = int(config.get('settings', 'jobs'))
        self.verbose = int(config.get('settings', 'verbose'))

        ## Load test parameters
        self.beam = int(config.get('test_params', 'beam'))
        self.lm_scale = float(config.get('test_params', 'lm_scale'))
        self.insertion_penalty = float(config.get('test_params', 'insertion_penalty'))

        ## Load pipeline
        self.test_pipeline = {}
        self.test_pipeline['coding'] = int(config.get('test_pipeline', 'coding'))
        self.test_pipeline['test'] = int(config.get('test_pipeline', 'test'))

        self.model = model
        self.mfc_list = '%s/mfc.list' %self.exp
        self.word_mlf = '%s/words.mlf' %self.exp
        self.phone_mlf = '%s/phones.mlf' %self.exp

        #self.lm = '/u/dgillick/workspace/hmm/wsj0/wdnet_bigram'
        self.lm = '%s/lm' %model.exp

        #self.dict = model.htk_dict
        self.dict = '%s/decode_dict' %model.exp
        #self.dict = '%s/wsj_dict_5k %model.common
        
        self.decode_func = 'hdecode'
        if self.decode_func == 'hvite': self.config_file = '%s/configcross' %model.common
        else: self.config_file = '%s/config.hdecode' %model.common
       
    def test(self, gaussians=1, iter=8, mmi=False, output_dir=None):

        ## Copy config file to the experiment dir
        config_output = '%s/config' %self.exp
        self.config.write(open(config_output, 'w'))
        log(self.logfh, 'TESTING with config [%s]' %config_output)

        if self.test_pipeline['coding']:
            import coding
            coding_dir = '%s/Coding' %self.exp
            util.create_new_dir(coding_dir)
            count = coding.wav_to_mfc(self, coding_dir, self.mfc_list)
            log(self.logfh, 'CODING finished [%d files]' %count)

        if self.test_pipeline['test']:
            import dict_and_lm
            start_time = time.time()
            num_utts, words = dict_and_lm.make_mlf_from_transcripts(model, self.dict, self.setup, self.data, self.word_mlf, self.mfc_list, skip_oov=True)
            log(self.logfh, 'wrote word mlf [%d utts] [%s]' %(num_utts, self.word_mlf))

            self.decode(model, self.mfc_list, self.word_mlf, self.lm, gaussians, iter, mmi, output_dir)
            total_time = time.time() - start_time
            log(self.logfh, 'TESTING finished; secs elapsed [%1.2f]' %total_time)

        
    def decode(self, model, mfc_list, gold_mlf, lm_file, gaussians, iter, mmi=False, output_dir=None):

        if mmi:
            model_file = '%s/MMI/Models/HMMI-%d-%d/MMF' %(model.exp, gaussians, iter)
        else:
            model_file = '%s/Xword/HMM-%d-%d/MMF' %(model.exp, gaussians, iter)
        model_list = '%s/tied.list' %model.exp

        if not output_dir: output_dir = '%s/decode' %self.exp
        output_dir = '%s/decode' %output_dir
        util.create_new_dir(output_dir)
        results_log = '%s/hresults.log' %output_dir
        output_mlf = '%s/decoded.mlf' %output_dir
    
        def hvite(input, output):
            cmd  = 'HVite -A -T 1 -l "*" -b silence '
            cmd += '-t %f ' %self.beam
            cmd += '-C %s ' %self.config_file
            cmd += '-H %s ' %model_file
            cmd += '-S %s ' %input
            cmd += '-i %s ' %output
            cmd += '-w %s ' %lm_file
            cmd += '-p %f ' %self.insertion_penalty
            cmd += '-s %f ' %self.lm_scale
            cmd += '%s %s' %(self.dict, model_list)
            return cmd

        ## HDecode parameters
        utts_per_split = 5
        block_size = 1
        word_end_beam = 150.0
        max_model = 0

        def hdecode(input, output):
            cmd  = 'HDecode -A -D -V -T 9 -o M -C %s' %self.config_file
            cmd += ' -H %s' %model_file
            cmd += ' -k %d' %block_size
            cmd += ' -t %f 100.0' %self.beam
            cmd += ' -v %f 115.0' %word_end_beam
            cmd += ' -u %d' %max_model
            cmd += ' -s %f' %self.lm_scale
            cmd += ' -p %f' %self.insertion_penalty
            cmd += ' -w %s' %lm_file
            cmd += ' -S %s' %input
            cmd += ' -i %s' %output
            cmd += ' %s %s' %(self.dict, model_list)
            if model.verbose > 0: cmd += ' >%s/%s.log' %(output_dir, os.path.basename(input))
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
            if self.decode_func == 'hvite':
                cmds.append(hvite(input, output))
            else:
                cmds.append(hdecode(input, output))

        if self.local == 1:
            for cmd in cmds:
                print cmd
                print os.popen(cmd).read()
        else:
            cmds_file = '%s/hvite.commands' %output_dir
            fh = open(cmds_file, 'w')
            for cmd in cmds: fh.write('%s\n' %cmd)
            fh.close()
            util.run_parallel(cmds_file, self.jobs, output_dir)
            #os.system('rm -f %s' %cmds_file)

        ## Merge outputs
        os.popen('rm -f %s' %output_mlf)
        os.popen('cat %s | grep -v "<" - > %s' %(' '.join(outputs), output_mlf))

        ## Evaluate
        cmd  = 'HResults -h -n -A -T 1 -c'
        cmd += ' -I %s' %gold_mlf
        cmd += ' %s %s > %s' %(model_list, output_mlf, results_log)
        os.system(cmd)
        print os.popen('cat ' + results_log).read()

        cmd = open(results_log).read().splitlines()[0]
        raw_wer = 100 - float(re.findall(r'Acc=([0-9.]*)', os.popen(cmd.replace('-h ', '')).read())[0].split('=')[-1])
        return raw_wer

        os.system('rm -f %s/mfc.list.* %s/align.output.*' %(output_dir, output_dir))


if __name__ == '__main__':

    from optparse import OptionParser
    usage = 'usage: %prog [options] model-config test-config'
    parser = OptionParser(usage=usage)
    parser.add_option('-j', '--jobs', dest='njobs', type=int, default=0,
                      help='number of run-command jobs to use')
    parser.add_option('-d', '--id', dest='exp_id', type=str, default='0',
                      help='experiment id')
    parser.add_option('-g', '--gaussians', dest='gaussians', type=int, default=1,
                       help='number of gaussian mixture components')
    parser.add_option('-i', '--iter', dest='iter', type=int, default=1,
                       help='iteration number')
    parser.add_option('-m', '--mmi', dest='mmi', default=False, action='store_true',
                       help='use MMI model')
    (options, args) = parser.parse_args()

    if len(args) < 2:
        sys.stderr.write('%s\n' %usage)
        sys.exit()

    import ConfigParser

    ## Read model config
    model_config = ConfigParser.ConfigParser()
    model_config.read(args[0])
    model = Model(model_config, train=False)

    ## Read decode config
    test_config = ConfigParser.ConfigParser()
    test_config.read(args[1])
    decoder = Decoder(test_config, model)

    if options.njobs > 0: decoder.jobs = options.njobs
    output_dir = '%s/decode-%s' %(decoder.exp, options.exp_id)
    decoder.test(options.gaussians, options.iter, options.mmi, output_dir)

