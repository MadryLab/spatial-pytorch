from argparse import ArgumentParser
import traceback
import os
import git
import torch as ch

import cox
import cox.utils
import cox.store

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers
    from . import defaults
    from .defaults import check_and_fill_args
except:
    print(traceback.format_exc())
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''
    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

    train_loader, val_loader = dataset.make_loaders(args.workers,
                    args.batch_size, data_aug=bool(args.data_aug))

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)

    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume)
    if 'module' in dir(model): model = model.module

    print(args)
    loader = (train_loader, val_loader)
    if not args.eval_only:
        model = train_model(args, model, loader, store=store)
    else:
        eval_model(args, val_loader, model, store)

    return model

def setup_args(args):
    '''
    Set a number of path related values in an arguments object. Also run the
    sanity check.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)
    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)
    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                        search_parent_directories=True)
    git_commit = repo.head.object.hexsha
    args.git_commit = git_commit

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.as_dict()
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)

    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)

    final_model = main(args, store=store)
