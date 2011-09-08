"""
General utilities
"""

import os, sys, time, re, gzip, cPickle

attr = ''
#attr += ' -attr \!squid7 -attr \!squid8 -attr \!squid9 -attr \!squid6 -attr \!squid5'
#attr += ' -attr Dual-Core-AMD-Opteron-Processor-875'
#attr +=  '-attr Intel-Xeon-X5550-@-2.67GHz'

def run(cmd, log_dir, my_attr=None):
   if my_attr == None: my_attr = attr
   rc_log = '%s/run-command_single.log' %log_dir
   os.system('run-command %s -attr noevict -log %s "%s"' %(my_attr, rc_log, cmd))
   return rc_log

def run_parallel(path, njobs, log_dir, my_attr=None):
   if my_attr == None: my_attr = attr
   rc_log = '%s/run-command.log' %log_dir
   cmd = 'run-command %s -attr noevict -J %d -f %s -log %s' %(my_attr, njobs, path, rc_log)
   #print cmd
   os.system(cmd)
   return rc_log
   
def save_pickle(data, path):
    print 'saving: %s' %path
    o = gzip.open(path, 'wb')
    cPickle.dump(data, o)
    o.close()

ZCAT = 'gzcat' if 'Darwin' in os.popen('uname -a').read().split() else 'zcat'
def load_pickle(path):
    #i = gzip.open(path, 'rb')
    i = os.popen(ZCAT + ' ' + path)
    data = cPickle.load(i)
    i.close()
    return data

def get_files(path, pattern):
   """
   Recursively find all files rooted in <path> that match the regexp <pattern>
   """
   L = []
   if not path.endswith('/'): path += '/'
   
   # base case: path is just a file
   if (re.match(pattern, os.path.basename(path)) != None) and os.path.isfile(path):
      L.append(path)
      return L
   
   # general case
   if not os.path.isdir(path):
      return L

   contents = os.listdir(path)
   for item in contents:
      item = path + item
      if (re.search(pattern, os.path.basename(item)) != None) and os.path.isfile(item):
         L.append(item)
      elif os.path.isdir(path):
         L.extend(get_files(item + '/', pattern))

   return L

def create_new_dir(dir):
   os.system('rm -rf %s' %dir)
   os.makedirs(dir)

def flush_file(fh):
   """
   flush file handle fh contents -- force write
   """
   fh.flush()
   os.fsync(fh.fileno())
   
def log_write(fh, line, extra=None):
   s = get_formatted_time() + '  ' + line + '\n'
   fh.write(s)
   sys.stderr.write(s)

def get_formatted_time():
    return time.strftime("%b %d %Y %H:%M:%S", time.localtime())

def exit(log):
   sys.stderr.write('Exiting at [%s]\nCheck log [%s]\n' %(get_formatted_time(), log))
   sys.exit()
