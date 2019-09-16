Creating and using custom datasets and models
=============================================

Creating a custom dataset
-------------------------
The robustness library by default includes most common datasets: ImageNet,
Restricted-ImageNet, CIFAR, CINIC, and A2B. That said, it is rather
straightforward to add your own dataset. 

1. Subclass the :py:class:`~robustness.datasets.DataSet` class in
   :mod:`robustness.datasets`. This means implementing 
   :py:meth:`~robustness.datasets.DataSet.__init__`
   and :py:meth:`~robustness.datasets.DataSet.get_model` functions.
2. In :samp:`__init__()`, all that is required is to call
   :samp:`super(NewClass, self).__init__` with the appropriate arguments,
   found in :py:class:`the docstring <robustness.datasets.DataSet>` and
   duplicated below:
   
   Arguments:
      - Dataset name (e.g. :samp:`imagenet`).
      - Dataset path (if your desired dataset is in the list of already implemented datasets in torchvision.datasets, pass the appropriate location, otherwise make this an argument of your subclassed :samp:`__init__` function.

     Named arguments (all required):
      - :samp:`num_classes`, the number of classes in the dataset
      - :samp:`mean`, the mean to normalize the dataset with
      - :samp:`std`, the standard deviation to normalize the dataset with
      - :samp:`custom_class`, the `torchvision.models` class corresponding
        to the dataset, if it exists (otherwise :samp:`None`)
      - :samp:`label_mapping`, a dictionary mapping from class numbers to
        human-interpretable class names (can be :samp:`None`)
      - :samp:`transform_train`, instance of :samp:`torchvision.transforms`
        to apply to the training images from the dataset
      - :samp:`transform_test`, instance of :samp:`torchvision.transforms`
        to apply to the validation images from the dataset
3. In :py:meth:`~robustness.datasets.DataSet.get_model`, implement a
   function which takes in an architecture name :samp:`arch` and boolean
   :samp:`pretrained`, and returns a PyTorch model (nn.Module) (see
   :py:meth:`the docstring <robustness.datasets.DataSet.get_model>` for
   more details). This will probably entail just using something like:::

      assert not pretrained, "pretrained only available for ImageNet"
      return models.__dict__[arch](num_classes=self.num_classes)
      # replace "models" with "cifar_models" in the above if the 
      # image size is less than [224, 224, 3]

4. Add an entry to :attr:`robustness.datasets.DATASETS` dictionary for your
   dataset.
5. If you want to be able to train a robust model on your dataset, add it
   to the :attr:`~robustness.main.DATASET_TO_CONFIG` dictionary in `main.py` and
   create a config file in the same manner as for the other datasets.

You're all set! You can create an instance of your dataset and a
corresponding model with:::

   from robustness.datasets import MyNewDataSet
   from robustness.model_utils import make_and_restore_model
   ds = MyNewDataSet('/path/to/dataset/')
   model, _ = make_and_restore_model(arch='resnet50', dataset=ds)

Creating a new architecture
----------------------------
Currently the robustness library supports a few common architectures. The
models are split between two folders: :samp:`cifar_models` for
architectures that handle CIFAR-size (i.e. 32x32x3) images, and
:samp:`imagenet_models` for models that require larger images (e.g.
224x224x3). It is possible to add architectures to either of these
folders, but to make them fully compatible with the :samp:`robustness`
library requires a few extra steps. 

We'll go through an example of how to add the AlexNet architecture for
ImageNet (just for the sake of illustration; note that this architecture is
already supported):
1. Download the `alexnet.py
<https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py>`
file containing the architecture specification (or write your own)
2. Add alexnet.py to the :samp:`imagenet_models` folder, and add to the
imports:::

      from .custom_modules import *

3. From :samp:`imagenet_models/__init__.py`, add the line:::
   
      from .alexnet import *

4. The AlexNet architecture is now available via:::

      from robustness.model_utils import make_and_restore_model
      from robustness.datasets import ImageNet
      ds = ImageNet('/path/to/imagenet')
      model, _ = make_and_restore_model(arch='alexnet', dataset=ds)

5. (If all you want to do with this architecture is training a robust
   model, **you can skip this step**). In order to make it fully compatible
   with the robustness library, the :samp:`forward` function of AlexNet
   must support the following three (boolean) options:

   - :samp:`with_latent` : If this option is given, :samp:`forward` should
     return the output of the second-last layer along with the logits.
   - :samp:`fake_relu` :  If this option is given, replace the ReLU just
     after the second-last layer with a :samp:`custom_modules.FakeReLUM`,
     which is a ReLU on the forwards pass and identity on the backwards
     pass.
   - :samp:`no_relu` : If this option is given, then :samp:`with_latent`
     should return the *pre-ReLU* activations of the second-last layer.

   These options are usually actually quite simple to implement: see
   the `AlexNet module <TODO>`_ for an example of how one might implement
   these three options.
