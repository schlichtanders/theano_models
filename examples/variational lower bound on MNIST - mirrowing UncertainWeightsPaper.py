from __future__ import division

import os
import numpy as np
import gzip
import cPickle
from climin.util import optimizer
from itertools import repeat, cycle, islice, izip
import random
inf = float("inf")

from breze.learn.data import one_hot
from breze.learn.base import cast_array_to_local_type
from schlichtanders.myfunctools import compose, meanmap, summap, compose_fmap, Average
from schlichtanders import myfunctools
from schlichtanders.mygenerators import eatN, chunk, chunk_list, every, takeN

from theano_models import (Merge, Flatten, Reparameterize, reduce_all_identities,
                           inputting_references, outputting_references)
from theano_models.tools import (as_tensor_variable, total_size, clone, clone_all,PooledRandomStreams,
                                 get_profile, squareplus, squareplus_inv, softplus, softplus_inv)
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
import theano_models.postmaps as post
from theano_models.composing import normalizing_flow, variational_bayes

from theano.tensor.shared_randomstreams import RandomStreams

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import warnings
from schlichtanders.myobjects import NestedNamespace
from schlichtanders.myos import replace_unc
import platform

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

pm.RNG = NestedNamespace(PooledRandomStreams(pool_size=int(5e8)), RandomStreams())
inputting_references.update(['to_be_randomized'])


# Data
# =====
datafile = os.path.join(__parent__, 'data', 'mnist.pkl.gz')

with gzip.open(datafile,'rb') as f:                                                                        
    train_set, val_set, test_set = cPickle.load(f)                                                       

X, Z = train_set                                                                                               
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

# from UncertainWeightsPaper rescaled the data,
# however it already seems to be normalized
# X /= 126  
# VX /= 126
# TX /= 126
# X *= 2
# VX *= 2
# TX *= 2

image_dims = 28, 28
X, Z, VX, VZ, TX, TZ = [cast_array_to_local_type(i) for i in (X, Z, VX,VZ, TX, TZ)]


# Hyperparameters
# ===============

engine = create_engine('sqlite:///' + os.path.join(__path__, 'hyperparameters.db'))
Base = declarative_base(bind=engine)


class RandomHyper(Base):
    __tablename__ = 'hyper'
    id = Column(Integer, primary_key=True)
    
    # hyper parameters:
    n_epochs = Column(Integer)
    batch_size = Column(Integer)
    mapreduce = Column(String)
    average_n = Column(Integer)
    units_per_layer = Column(Integer)
    pi = Column(Float)
    minus_log_s1 = Column(Integer)
    minus_log_s2 = Column(Integer)
    
    n_normflows = Column(Integer)
    
    opt_identifier = Column(String)
    opt_momentum = Column(Float)
    opt_offset = Column(Float)
    opt_decay = Column(Float)
    opt_step_rate = Column(Float)
    
    # results:
    best_val_loss = Column(Float)
    best_parameters = Column(PickleType, nullable=True)
    train_loss = Column(PickleType)
    val_loss = Column(PickleType)

    def __init__(self):
        # hyper parameters:
        self.n_epochs = 20
        self.batch_size = 128
        self.mapreduce = random.choice(["summap", "meanmap"])
        self.average_n = 1
        self.units_per_layer = np.random.choice([400, 800, 1200], p=[0.5, 0.3, 0.2])
        self.pi = random.choice([1/4, 1/2, 3/4])
        self.minus_log_s1 = random.choice([0,1,2])
        self.minus_log_s2 = random.choice([6,7,8])
        
        self.n_normflows = random.choice([1,2,3,4,8,32])
        
        self.opt_identifier = random.choice(["adadelta", "adam", "rmsprop"])
        if self.opt_identifier == "adadelta":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.9, 1)])
            self.opt_offset = random.choice([5e-5, 1e-8])
        elif self.opt_identifier == "adam":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.8, 0.93)])
            self.opt_offset = 10 ** -np.random.uniform(3, 4)
        elif self.opt_identifier == "rmsprop":
            self.opt_momentum = random.choice([np.random.uniform(0.002, 0.008), np.random.uniform(0.9, 1)]),
            self.opt_offset = np.random.uniform(0, 0.000045)
        self.opt_decay = np.random.uniform(0.78, 1)
        self.opt_step_rate = random.choice([1e-3, 1e-4, 1e-5])
        
        self.init_results()
    
    def init_results(self):
        # extra for being able to reset results for loaded hyperparameters
        self.best_parameters = None
        self.best_val_loss = inf
        self.train_loss = []
        self.val_loss = []

Base.metadata.create_all()
Session = sessionmaker(bind=engine)
sql_session = Session()
hyper = RandomHyper()


# Model
# =====

# data modelling
# --------------

# this is extremely useful to tell everything the default sizes
input = as_tensor_variable(X[0], name="X")

predictor = dm.Mlp(
    input=input,
    output_size=Z.shape[1],
    output_transfer="softmax",
    hidden_sizes=[hyper.units_per_layer]*2,
    hidden_transfers=["rectifier"]*2
)
target_distribution = pm.Categorical(predictor)
targets = Merge(target_distribution, predictor,
                Flatten(predictor['parameters'], flat_key="to_be_randomized")) #givens={predictor['inputs'][0]: X[0]}


# parameter modelling
# -------------------

params_base = pm.DiagGauss(output_size=total_size(targets['to_be_randomized']))
normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]

params = params_base
for transform in normflows:
    params = normalizing_flow(transform, params)  # returns transform, however with adapted logP    


# bayes
# -----

g1 = pm.Gauss(total_size(targets['to_be_randomized']), init_var=np.exp(-2* hyper.minus_log_s1))
g2 = pm.Gauss(total_size(targets['to_be_randomized']), init_var=np.exp(-2* hyper.minus_log_s2))
prior = pm.Mixture(g1, g2, mixture_probs=[hyper.pi, 1-hyper.pi])
# label hyper parameters accordingly
prior = Merge(prior,
              parameters=None, # mean is not adapted at all, but left centred at zero
              parameters_positive='hyperparameters_positive',
              parameters_psumto1='hyperparameters_psumto1')
model = variational_bayes(targets, 'to_be_randomized', params, priors=prior)
model = Merge(model, Reparameterize(model['parameters_positive'], softplus, softplus_inv))  # softplus used in the paper
model = Merge(model, Flatten(model['parameters']))


# Optimizer
# =========

reduce_all_identities()

n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

def weights_regularizer_1epoch():
    for i in range(1, n_batches+1):
        yield 2**(n_batches - i) / (2**n_batches - 1)
        
assert len(list(weights_regularizer_1epoch())) == n_batches


mapreduce = getattr(myfunctools, hyper.mapreduce)
if hyper.average_n > 1:
    mapreduce = compose_fmap(Average(hyper.average_n), mapreduce)

postmap = compose(post.flat_numericalize_postmap, post.variational_postmap)
postmap_kwargs = {
    'mapreduce': mapreduce,  # TODO add more functionality for composed fmaps, with args
    'annealing_combiner': post.AnnealingCombiner(
        weights_regularizer=cycle(weights_regularizer_1epoch())
    ),
    'adapt_init_params': lambda ps: ps + np.random.normal(size=ps.size, scale=0.01),
    'mode': 'FAST_RUN'
}
optimizer_kwargs = postmap(model, **postmap_kwargs)
climin_kwargs = post.climin_postmap(optimizer_kwargs)

opt = optimizer(
    identifier=hyper.opt_identifier,
    step_rate=hyper.opt_step_rate,
    momentum=hyper.opt_momentum,
    decay=hyper.opt_decay,
    offset=hyper.opt_offset,
    
    args=climin_args,
    **climin_kwargs
)


# Fit
# ===

for info in takeN(hyper.n_epochs, every(n_batches, opt)):
    current_epoch = info['n_iter']//n_batches
    # collect and visualize validation loss for choosing the best model
    val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)
    if val_loss < hyper.best_val_loss:
        hyper.best_parameters = opt.wrt
        hyper.best_val_loss = val_loss
    hyper.val_loss.append(val_loss)
    
    # visualize training loss for comparison:
    training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
    hyper.train_loss.append(training_loss)

# Results
# =======

# just save them
sql_session.add(hyper)
sql_session.commit()