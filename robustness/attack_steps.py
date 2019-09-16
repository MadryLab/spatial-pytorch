import torch as ch
import itertools
from . import spatial
class AttackerStep:
    '''
    Generic class for attacker steps, under perturbation constraints
    specified by an "origin input" and a perturbation magnitude.
    Must implement project, step, and random_perturb
    '''
    def __init__(self, orig_input, eps, step_size, use_grad=True):
        '''
        Initialize the attacker step with a given perturbation magnitude.

        Args:
            eps (float): the perturbation magnitude
            orig_input (ch.tensor): the original input
        '''
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
        self.use_grad = use_grad

    def project(self, x):
        '''
        Given an input x, project it back into the feasible set

        Args:
            ch.tensor x : the input to project back into the feasible set.

        Returns:
            A `ch.tensor` that is the input projected back into
            the feasible set, that is,
        .. math:: \min_{x' \in S} \|x' - x\|_2
        '''
        raise NotImplementedError

    def step(self, x, g):
        '''
        Given a gradient, make the appropriate step according to the
        perturbation constraint (e.g. dual norm maximization for :math:`\ell_p` norms).

        Parameters:
            g (ch.tensor): the raw gradient

        Returns:
            The new input, a ch.tensor for the next step.
        '''
        raise NotImplementedError

    def random_perturb(self, x):
        '''
        Given a starting input, take a random step within the feasible set
        '''
        raise NotImplementedError

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_\infty \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def step(self, x, g):
        """
        """
        step = ch.sign(g) * self.step_size
        return x + step

    def random_perturb(self, x):
        """
        """
        return 2 * (ch.rand_like(x) - 0.5) * self.eps

# L2 threat model
class L2Step(AttackerStep):
    """
    Attack step for :math:`\ell_\infty` threat model. Given :math:`x_0`
    and :math:`\epsilon`, the constraint set is given by:

    .. math:: S = \{x | \|x - x_0\|_2 \leq \epsilon\}
    """
    def project(self, x):
        """
        """
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.orig_input + diff

    def step(self, x, g):
        """
        """
        # Scale g so that each element of the batch is at least norm 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        return x + scaled_g * self.step_size

    def random_perturb(self, x):
        """
        """
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)

NUM_TRANS = 5
NUM_ROT = 31

class Spatial:
    def __init__(self, attack_type, spatial_constraint):
        self.use_grad = False
        self.rot_constraint = float(spatial_constraint)

        self.trans_constraint = (self.rot_constraint/10.)/28.
        self.attack_type = attack_type

    def project(self, x):
        return x

    def random_perturb(self, x):
        return x

    def step(self, x, g, correcter=None):
        assert x.shape[2] == x.shape[3]
        max_trans = self.trans_constraint
        max_rot = self.rot_constraint

        bs = x.shape[0]

        device = x.get_device()
        if self.attack_type == 'random':
            rots = spatial.unif((bs,), -max_rot, max_rot)
            txs = spatial.unif((bs, 2), -max_trans, max_trans)

            transformed = transform(x, rots, txs)
            return transformed

        assert self.attack_type == 'grid'
        assert x.shape[0] == 1

        rots = ch.linspace(-max_rot, max_rot, steps=NUM_ROT)
        trans = ch.linspace(-max_trans, max_trans, steps=NUM_TRANS)
        tfms = ch.tensor(list(itertools.product(rots, trans, trans))).cuda(device=device)

        all_rots = tfms[:, 0]
        all_trans = tfms[:, 1:]

        ntfm = all_rots.shape[0]
        transformed = transform(x.repeat([ntfm, 1, 1, 1]), all_rots, all_trans)

        i = 0
        all_losses = []
        while i < ntfm:
            to_do = transformed[i:i+MAX_BS]
            is_correct = correcter(to_do)
            argmin = is_correct.argmin()
            if is_correct[argmin] == 0:
                return transformed[i+argmin:i+argmin+1]

            i += MAX_BS

        return transformed[0:1]

MAX_BS = 200
# x: [bs, 3, w, h]
# rotation: [bs]
# translation: [bs, 2]
# uses bilinear
def transform(x, rotation, translation):
    assert x.shape[1] == 3

    with ch.no_grad():
        translated = spatial.transform(x, rotation, translation)
        #    rotated = kornia.rotate(x, rotation)
        #    translated = kornia.translate(rotated, translation)

    return translated

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    """
    Unconstrained threat model, :math:`S = \mathbb{R}^n`.
    """
    def project(self, x):
        """
        """
        return x

    def step(self, x, g):
        """
        """
        return x + g * self.step_size

    def random_perturb(self, x):
        """
        """
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=step_size)
