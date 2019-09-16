import argparse
import shutil

from subprocess import Popen, PIPE

arg_names = ['orig_attack', 'orig_spatial_constraint', 'orig_out_dir',
                       'eval_out_dir', 'eval_attack']

def execute(cmd):
    process = Popen(cmd, stdout=PIPE, shell=True)
    (output, err) = process.communicate()
    return process.wait()

def main(args):
    cmd = ' '.join(getattr(args, k) for k in arg_names)
    cmd = './eval.sh ' + cmd
    execute(cmd)

