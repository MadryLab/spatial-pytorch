from gridder.generator import PATH_KEY

NUM_SESSIONS = 1
PARAMS = {
    'orig_attack':['nocrop', 'standard', 'random', 'worst10'],
    'orig_spatial_constraint': ['30', '40'],
    'orig_out_dir':['/data/theory/robustopt/engstrom/store/spatial'],
    'eval_out_dir':['/data/theory/robustopt/engstrom/store/spatial_eval'],
    'eval_attack':['random', 'worst10', 'grid']
}

GPUS_PER_JOB = 8
MAX_JOBS_PER_GPU = 1
PROJECT_LOCATION = '/data/theory/robustopt/engstrom/src/spatial-pytorch'
PATH_TO_MAIN = 'eval.py'
LOCATION_IS_GIT = False

def rule(p):
    sc = p['orig_spatial_constraint']
    oa = p['orig_attack']
    if oa in ['nocrop', 'standard'] and sc in ['30']:
        return False

    return True

RULES = [rule]

