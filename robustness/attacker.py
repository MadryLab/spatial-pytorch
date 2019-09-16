import torch as ch
import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from .tools import helpers
from . import attack_steps

STEPS = {
    'spatial': attack_steps.Spatial,
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep
}

class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, tries, spatial_constraint, attack_type,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint ("2"|"inf"|"unconstrained") : threat model for
                adversarial attacks (:math:`\ell_2` ball, :math:`\ell_\infty` ball,
                :math:`\mathbb{R}^n`.
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            tries (int): number of times to try adv attack
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.

        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)
        """

        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()

        step = STEPS['spatial'](attack_type=attack_type,
                                spatial_constraint=spatial_constraint)

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output

        # Main function for making adversarial examples
        def get_adv_examples(x):
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = ch.clamp(x + step.random_perturb(x), 0, 1)

            iterator = range(tries)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = losses.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            # PGD iterates
            with ch.no_grad():
                # def special_losser(x):
                #     targets = target.repeat([x.shape[0]])
                #     return calc_loss(x, targets)[0]

                ##### for spatial attacker
                def calc_correct(inp):
                    if should_normalize:
                        inp = self.normalize(inp)

                    output = self.model(inp)
                    targets = target.repeat([inp.shape[0]])
                    return output.argmax(dim=1) == targets

                #####

                # set initial losses
                if use_best:
                    losses, _ = calc_loss(x, target)
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args)

                for _ in iterator:
                    adv_x = step.step(x, None, correcter=calc_correct)
                    adv_x = ch.clamp(adv_x, 0, 1)
                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

                    losses, out = calc_loss(adv_x, target)
                    assert losses.shape[0] == adv_x.shape[0], 'Shape of losses must match input!'
                    loss = ch.mean(losses)

                    args = [losses, best_loss, adv_x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, adv_x)

            return best_x

        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv = get_adv_examples(orig_cpy)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModel(ch.nn.Module):
    """
    Wrapper class for adversarial attacks on models. Given any normal
    model (a `ch.nn.Module` instance), wrapping it in AttackerModel allows
    for convenient access to adversarial attacks and other applications.

    >>> model = ResNet50()
    >>> model = AttackerModel(model)
    >>> x = ch.rand(10, 3, 32, 32) # random images
    >>> y = ch.zeros(10) # label 0
    >>> out, new_im = model(x, y, make_adv=True) # adversarial attack
    >>> out, new_im = model(x, y, make_adv=True, targeted=True) # targeted attack
    >>> out = model(x) # normal inference (no label needed)

    More code examples available in the documentation for `forward`.
    For a more comprehensive overview of this class, see `our documentation <TODO>`_
    """
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, repeat=True,
                **attacker_kwargs):
        """
        Main function for running inference and generating adversarial
        examples for a model.

        Parameters:
            inp (ch.tensor) : input to do inference on [N x input_shape] (e.g. NCHW)
            target (ch.tensor) : ignored if `make_adv == False`. Otherwise,
                labels for adversarial attack.
            make_adv (bool) : whether to make an adversarial example for
                the model. If true, returns a tuple of the form
                :samp:`(model_prediction, adv_input)` where
                :samp:`model_prediction` is a tensor with the *logits* from
                the network.
            with_latent (bool) : also return the second-last layer along
                with the logits. Output becomes of the form
                :samp:`((model_logits, model_layer), adv_input)` if
                :samp:`make_adv==True`, otherwise :samp:`(model_logits, model_layer)`.
            fake_relu (bool) : useful for activation maximization. If
                :samp:`True`, replace the ReLUs in the last layer with
                "fake ReLUs," which are ReLUs in the forwards pass but
                identity in the backwards pass (otherwise, maximizing a
                ReLU which is dead is impossible as there is no gradient).
            no_relu (bool) : If :samp:`True`, return the latent output with 
                the (pre-ReLU) output of the second-last layer, instead of the
                post-ReLU output. Requires :samp:`fake_relu=False`, and has no 
                visible effect without :samp:`with_latent=True`.
            with_image (bool) : if :samp:`False`, only return the model output
                (even if :samp:`make_adv == True`).

        """
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        if with_image:
            normalized_inp = self.normalizer(inp)

            if no_relu and (not with_latent):
                print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
            if no_relu and fake_relu:
                raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

            output = self.model(normalized_inp, with_latent=with_latent,
                                    fake_relu=fake_relu, no_relu=no_relu)
        else:
            raise ValueError("what")
            output = None

        return (output, inp)
