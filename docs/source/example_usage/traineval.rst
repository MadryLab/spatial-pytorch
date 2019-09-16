Training and evaluating networks
================================

In this walkthrough, we'll go over how to train and evaluate networks via the
:mod:`robustness.main` command-line tool.

Training a standard (nonrobust) model
--------------------------------------
We'll start by training a standard (non-robust) model. This is accomplished through the following command:

.. code-block:: bash

   python -m robustness.main --dataset DATASET --data /path/to/dataset \
      --adv-train 0 --arch ARCH --out-dir /logs/checkpoints/dir/

In the above, :samp:`DATASET` can be any supported dataset (i.e. in
:attr:`robustness.datasets.DATASETS`). For a demonstration of how to add a
supported dataset, see :doc:`datasets`.

With the above command, you should start seeing progress bars indicating that
the training has begun! Note that there are a whole host of arguments that you
can customize in training, including optimizer parameters (e.g. :samp:`--lr`,
:samp:`--weight-decay`, :samp:`--momentum`), logging parameters (e.g.
:samp:`--log-iters`, :samp:`--save-ckpt-iters`), and learning rate schedule. To
see more about these arguments, we run:

.. code-block:: bash

   python -m robustness --help

For completeness, the full list of parameters related to *non-robust* training
are below:

.. code-block:: bash

     --out-dir OUT_DIR     where to save training logs and checkpoints (default:
                        required)
     --config-path CONFIG_PATH
                           config path for loading in parameters (default: None)
     --exp-name EXP_NAME   where to save in (inside out_dir) (default: None)
     --dataset {imagenet,restricted_imagenet,cifar,cinic,a2b}
                           (choices: {arg_type}, default: required)
     --data DATA           path to the dataset (default: /tmp/)
     --arch ARCH           architecture (see {cifar,imagenet}_models/ (default:
                           required)
     --batch-size BATCH_SIZE
                           batch size for data loading (default: by dataset)
     --workers WORKERS     # data loading workers (default: 30)
     --resume RESUME       path to checkpoint to resume from (default: None)
     --data-aug {0,1}      whether to use data augmentation (choices: {arg_type},
                           default: 1)
     --epochs EPOCHS       number of epochs to train for (default: by dataset)
     --lr LR               initial learning rate for training (default: 0.1)
     --weight_decay WEIGHT_DECAY
                           SGD weight decay parameter (default: by dataset)
     --momentum MOMENTUM   SGD momentum parameter (default: 0.9)
     --step-lr STEP_LR     number of steps between 10x LR drops (default: by
                           dataset)
     --custom-schedule CUSTOM_SCHEDULE
                           LR sched (format: [(epoch, LR),...]) (default: None)
     --log-iters LOG_ITERS
                           how frequently (in epochs) to log (default: 5)
     --save-ckpt-iters SAVE_CKPT_ITERS
                           how frequently (epochs) to save (-1 for bash, only
                           saves best and last) (default: -1)

Finally, there is one additional argument, :samp:`--adv-eval {0,1}`, that enables
adversarial evaluation of the non-robust model as it is being trained (i.e.
instead of reporting just standard accuracy every few epochs, we'll also report
robust accuracy if :samp:`--adv-eval 1` is added). However, adding this argument
also necessitates the addition of hyperparameters for adversarial attack, which
we cover in the following section.

Training a robust model (adversarial training)
--------------------------------------------------
To train a robust model we proceed in the exact same way as for a standard
model, but with a few changes. First, we change :samp:`--adv-train 0` to
:samp:`--adv-train 1` in the training command. Then, we need to make sure to
supply all the necessary hyperparameters for the attack:

.. code-block:: bash

     --attack-steps ATTACK_STEPS
                        number of steps for adversarial attack (default: 7)
     --constraint {inf,2,unconstrained}
                           adv constraint (choices: {arg_type}, default:
                           required)
     --eps EPS             adversarial perturbation budget (default: required)
     --attack-lr ATTACK_LR
                           step size for PGD (default: required)
     --use-best {0,1}      if 1 (0) use best (final) PGD step as example
                           (choices: {arg_type}, default: 1)
     --random-restarts RANDOM_RESTARTS
                           number of random PGD restarts for eval (default: 0)
     --eps-fadein-epochs EPS_FADEIN_EPOCHS
                           fade in eps over this many iterations (default: 0)


Examples
--------
Training a non-robust ResNet-18 for the CIFAR dataset:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m robustness.main --dataset cifar --data /path/to/cifar \
      --adv-train 0 --arch resnet18 --out-dir /logs/checkpoints/dir/

Training a robust ResNet-50 for the Restricted-ImageNet dataset:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m robustness.main --dataset restricted_imagenet --data \
      /path/to/imagenet --adv-train 1 --arch resnet50 \
      --out-dir /logs/checkpoints/dir/ --eps 3.0 --attack-lr 0.5 \
      --attack-steps 7 --constraint 2

Reading and analyzing training results
--------------------------------------
By default, the above command will create a folder in
:samp:`/logs/checkpoints/dir/` with a random uuid for a name (you can set this
name manually via the :samp:`--exp-name` argument). At the end of training, the
folder structure will look like:::

   /logs/checkpoints/dir/RANDOM_UUID
      checkpoint.latest.pt
      checkpoint.best.pt
      store.h5
      tensorboard/
      save/ 
