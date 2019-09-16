Using robustness as a general training library
==============================================
In the other walkthroughs, we've demonstrated how to use :samp:`robustness` as
a :doc:`Command line tool for training and evaluating models <traineval>`, and
how to use it as a library for :doc:`customloss`. Here, we'll demonstrate how
:samp:`robustness` can be used a general library for experimenting with neural
network training.

In this document, we'll walk through using robustness as a library from another
separate project :samp:`my-training-experiments`. We'll start by making a
`main.py` file. *We strongly recommend just copying the source of :mod:`robustness.main`,
which can be found `here <TODO>`_, as it will get us started with a fully
configured argument parser, logging framework, and imports.* That said, if you
don't want the full flexibility of all of those arguments, our main file can be
very bare-boned. For example the following :samp:`main.py` file suffices for
training an adversarially robust CIFAR classifier with a fixed set of
parameters:

.. code-block:: python

   from robustness import model_utils, datasets, train, defaults
   from robustness.datasets import CIFAR
   import torch as ch
   from cox.utils import Parameters
   import cox.store

   # Hard-coded dataset, architecture, batch size, workers
   ds = CIFAR('/tmp/')
   m, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
   train_loader, val_loader = ds.make_loaders(batch_size=128, workers=8)

   # Create a cox store for logging
   out_store = cox.store.Store(OUT_DIR)

   # Hard-coded base parameters
   train_kwargs = {
       'out_dir': "train_out",
       'adv_train': 1,
       'constraint': '2',
       'eps': 0.5,
       'attack_lr': 1.5,
       'attack_steps': 20
   }
   train_args = Parameters(train_kwargs)

   # Fill whatever parameters are missing from the defaults
   train_args = defaults.check_and_fill_args(train_args,
                           defaults.TRAINING_ARGS, CIFAR)
   train_args = defaults.check_and_fill_args(train_args,
                           defaults.PGD_ARGS, CIFAR)

   # Train a model
   train.train_model(train_args, m, (train_loader, val_loader), store=out_store)

For the sake of space, we'll continue with this bare-bones example, but
everything that we show here can just as easily be applied to a copied
:mod:`robustness.main` file.

Training networks with custom loss functions
--------------------------------------------
By default, training uses the cross-entropy loss; however, we can easily change
this by specifying a custom training loss and a custom adversary loss. For
example, suppose that instead of just computing the cross-entropy loss, we're
going to try an experimental new training loss that multiplies a random 50%
of the logits by 10. (*Note that this is just for illustrative purposes---in
practice this is a terrible idea*.)

We can implement this crazy loss function as a training criterion and a
corresponding adversary loss. Recall that as discussed in the
:meth:`robustness.train.train_model` docstring, the train loss takes in
:samp:`logits,targets` and returns a scalar, whereas the adversary loss takes in
:samp:`model,inputs,targets` and returns a vector (not averaged along the
batch) as well as the output.

.. code-block:: python

   train_crit = ch.nn.CrossEntropyLoss()
   def custom_train_loss(logits, targ):
       probs = ch.ones_like(logits) * 0.5
       logits_to_multiply = ch.bernoulli(probs) * 9 + 1
       return train_crit(logits_to_multiply * logits, targ)
       
   adv_crit = ch.nn.CrossEntropyLoss(reduction='none').cuda()
   def custom_adv_loss(model, inp, targ):
       logits = model(inp)
       probs = ch.ones_like(logits) * 0.5
       logits_to_multiply = ch.bernoulli(probs) * 9 + 1
       new_logits = logits_to_multiply * logits
       return adv_crit(new_logits, targ), new_logits

   train_kwargs['custom_train_loss'] = custom_train_loss
   train_kwargs['custom_adv_loss'] = custom_adv_loss

Adding these few lines right after the declaration of :samp:`train_kwargs`
suffices for training our network robustly with this custom loss.

Training networks with custom data loaders
-------------------------------------------
Another aspect of the training we can customize is data loading, through two
utilities for modifying dataloaders called
:meth:`robustness.loaders.TransformedLoader` and
:class:`robustness.loaders.LambdaLoader`. To see how they work, we're going to
consider two variations on our training: (a) training with label noise, and (b)
training with random labels.

Using LambdaLoader to train with label noise
""""""""""""""""""""""""""""""""""""""""""""
:class:`~robustness.laoders.LambdaLoader` works by modifying the output of a
data loader *in real-time*, i.e. it applies a fixed function to the output of a
loader. This makes it well-suited to, e.g., custom data augmentation,
input/label noise, or other applications where randomness across batches is
needed. To demonstrate its usage, we're going to add label noise to our training
setup. To do this, all we need to do is define a function which takes in a batch
of inputs and labels, and returns the same batch but with label noise added in.
For example:

.. code-block:: python

   from robustness.loaders import LambdaLoader

   def label_noiser(ims, labels):
       label_noise = ch.randint_like(labels, high=9)
       probs = ch.ones_like(logits) * 0.1
       labels_to_noise = ch.bernoulli(probs.float()).long()
       new_labels = (labels + label_noise * labels_to_noise) % 10
       return ims, new_labels

   train_loader = LambdaLoader(train_loader, label_noiser)

Note that LamdaLoader is quite general---any function that takes in :samp:`ims,
labels` and outputs :samp:`ims, labels` of the same shape can be put in place of
:samp:`label_noiser` above.

Using TransformedLoader to train with random labels
"""""""""""""""""""""""""""""""""""""""""""""""""""
In contrast to :class:`~robustness.loaders.LambdaLoader`,
:meth:`~robustness.loaders.TransformedLoader` is a data loader transformation
that is applied *once* at the beginning of training (this makes it better suited
to deterministic transformations to inputs or labels). Unfortunately, the
implementation of TransformedLoader currently loads the entire dataset into
memory, so it only reliably works on small datasets (e.g. CIFAR). This will be 
fixed in a future version of the library. To demonstrate its usage, we will use 
it to randomize labels for the training set. (Recall that when we usually train
using random labels, we perform the label assignment only once, prior to 
training.) To do this, all we need to do is define a function which takes in a
batch of inputs and labels, and returns the same batch, but with random labels
instead. For example:

.. code-block:: python

   from robustness.loaders import TransformedLoader
   from robustness.data_augmentation import TRAIN_TRANSFORMS_DEFAULT

   train_loader, val_loader = ds.make_loaders(workers=NUM_WORKERS, 
                                                   batch_size=BATCH_SIZE,
                                                   data_aug=False)

  def rand_label_transform(nclasses):
      def make_rand_labels(ims, targs):
          new_targs = ch.randint(0, high=nclasses,size=targs.shape).long()      
          return ims, new_targs
      return make_rand_labels

   train_loader_transformed = TransformedLoader(train_loader,
                                              rand_label_transform(10),
                                              TRAIN_TRANSFORMS_DEFAULT(32), 
                                              workers=NUM_WORKERS, 
                                              batch_size=BATCH_SIZE,
                                              do_tqdm=True)

Here, we start with a :samp:`train_loader` without data augmentation, to get access 
to the actual image-label pairs from the training set. We then transform each input
by assigning an image a random label instead. Moreover, we also support applying other
transforms in *real-time* (such as data augmentation) during the creation of the 
transformed dataset using :samp:`train_loader_transformed` (e.g., 
:samp:`TRAIN_TRANSFORMS(32)` here).

Note that TransformedLoader is quite general---any function that takes in :samp:`ims,
labels` and outputs :samp:`ims, labels` of the same shape can be put in place of
:samp:`rand_label_transform` above. 

Training networks with custom logging
-------------------------------------
The library also supports training with custom logging functionality, either applied
every epoch or iteration/step. Here, we demonstrate this functionality using a 
logging function that measures the norm of the network parameters (by treating 
them as a single vector). We will modify/augment the :samp:`main.py`code described 
above:

.. code-block:: python

   from torch.nn.utils import parameters_to_vector as flatten

   def norm_log_hook(out_table):
    def log_norm(mod, log_info):
          curr_params = flatten(mod.parameters())
          log_info_custom = {'epoch': log_info['epoch'],
                             'weight_norm': ch.norm(curr_params).detach().cpu().numpy()
                             }
          out_table.append_row(log_info_custom)
    return log_norm

We must now create a custom cox store `here <TODO>`_  for logging.

.. code-block:: python

    import cox.store

    out_store = cox.store.Store(OUT_DIR)
    CUSTOM_SCHEMA = {'epoch': int, 
                     'weight_norm': float
                     }

    out_store.add_table('custom', CUSTOM_SCHEMA)

We will then modify the :samp:`train_kwargs` to incorporate this function into 
the logging done per epoch/iteration. If we want to log the norm of the weights
every epoch, we can do:

.. code-block:: python

  train_kwargs['epoch_hook'] = norm_log_hook(out_store['custom'])

If we want to perform the logging every iteration, we need to make the
following modifications:

.. code-block:: python

  CUSTOM_SCHEMA = {'iteration': int, 
                   'weight_norm': float
                   }
  out_store.add_table('custom', CUSTOM_SCHEMA)

  def norm_log_hook(out_table):
    def log_norm(mod, it, loop_type, inp, targ):
      if loop_type == 'train':
        curr_params = flatten(mod.parameters())
        log_info_custom = {'iteration': it,
                           'weight_norm': ch.norm(curr_params).detach().cpu().numpy()
                           }
        out_table.append_row(log_info_custom)
    return log_norm

  train_kwargs['iteration_hook'] = norm_log_hook(out_store['custom'])

Note that we need to change the hook function to take the correct number of inputs.