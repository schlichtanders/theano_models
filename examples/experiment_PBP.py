# coding: utf-8
from __future__ import division

import os, platform, sys, traceback
from pprint import pformat, pprint
import numpy as np
from climin.util import optimizer
from itertools import repeat, cycle, islice, izip
import random
inf = float("inf")

from schlichtanders.myfunctools import compose, meanmap, summap, compose_fmap, Average
from schlichtanders.mygenerators import eatN, chunk, chunk_list, every, takeN

import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
from theano_models import data

from sklearn import cross_validation
from theano.tensor.shared_randomstreams import RandomStreams

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

tm.inputting_references.update(['to_be_randomized'])
tm.inputting_references, tm.outputting_references

from schlichtanders.myobjects import NestedNamespace
pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(5e8)), RandomStreams())

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

suffix = "_"+sys.argv[1] if len(sys.argv) > 1 else "_several"

# # Data
#     # datasetnames = ["boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "protein", "winered", "yacht", "year"]
#     datasetnames = ["boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "winered", "yacht"]
datasetname = "concrete"

Z, X = getattr(data, "_" + datasetname)()
# normalization is standard in Probabilistic Backpropagation Paper
X_mean = X.mean(0)
X_std = X.std(0)
X = (X - X_mean) / X_std
Z_mean = Z.mean(0)
Z_std = Z.std(0)
Z = (Z - Z_mean) / Z_std

X, TX, Z, TZ = cross_validation.train_test_split(X, Z, test_size=0.1) # 10% test used in paper
X, VX, Z, VZ = cross_validation.train_test_split(X, Z, test_size=0.1) # 20% validation used in paper


# # Hyperparameters

engine = create_engine('sqlite:///' + os.path.join(__path__, 'hyperparameters2%s.db' % suffix))
Base = declarative_base(bind=engine)


class RandomHyper(Base):
    __tablename__ = "hyper"
    id = Column(Integer, primary_key=True)
    
    # hyper parameters:
    datasetname = Column(String)
    max_epochs_without_improvement = Column(Integer)
    average_n = Column(Integer)
    units_per_layer = Column(Integer)
    minus_log_s = Column(Integer)
    batch_size = Column(Integer)
    
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

    # alternative:
    alternative_best_val_loss = Column(Float)
    alternative_best_parameters = Column(PickleType, nullable=True)
    alternative_train_loss = Column(PickleType)
    alternative_val_loss = Column(PickleType)

    # baseline:
    baseline_best_val_loss = Column(Float)
    baseline_best_parameters = Column(PickleType, nullable=True)
    baseline_train_loss = Column(PickleType)
    baseline_val_loss = Column(PickleType)

    def __init__(self):
        self.datasetname = datasetname
        # hyper parameters:
        self.max_epochs_without_improvement = 30
        self.batch_size = random.choice([1,10, 100])
        self.average_n = 1
        self.units_per_layer = 50
        self.minus_log_s = random.choice([1,2,3,4,5,6,7])
        # the prior is learned together with the other models in analogy to the paper Probabilistic Backpropagation
        
        self.n_normflows = random.choice([1,2,3,4,8,32])
        
        self.opt_identifier = random.choice(["adadelta", "adam", "rmsprop"])
        if self.opt_identifier == "adadelta":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.9, 1)])
            self.opt_offset = random.choice([5e-5, 1e-8])
            self.opt_step_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        elif self.opt_identifier == "adam":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.8, 0.93)])
            self.opt_offset = 10 ** -np.random.uniform(3, 4)
            self.opt_step_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        elif self.opt_identifier == "rmsprop":
            self.opt_momentum = random.choice([np.random.uniform(0.002, 0.008), np.random.uniform(0.9, 1)])
            self.opt_offset = np.random.uniform(0, 0.000045)
            self.opt_step_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        self.opt_decay = np.random.uniform(0.78, 1)
        
        self.init_results()
    
    def init_results(self):
        # extra for being able to reset results for loaded hyperparameters
        self.best_parameters = None
        self.best_val_loss = inf
        self.train_loss = []
        self.val_loss = []

        self.alternative_best_val_loss = inf
        self.alternative_best_parameters = None
        self.alternative_train_loss = []
        self.alternative_val_loss = []

        self.baseline_best_val_loss = inf
        self.baseline_best_parameters = None
        self.baseline_train_loss = []
        self.baseline_val_loss = []

Base.metadata.create_all()
Session = sessionmaker(bind=engine)
sql_session = Session()



# Main Loop
# =========
while True:
    try:
        hyper = RandomHyper()
        sql_session.add(hyper)
        pprint(hyper.__dict__)
        # reset hard the saved models:
        dm.InvertibleModel.INVERTIBLE_MODELS = []
        tm.Model.all_models = []


        # Model Normalizing flow
        # ======================

        # this is extremely useful to tell everything the default sizes
        input = tm.as_tensor_variable(X[0], name="X")

        predictor = dm.Mlp(
            input=input,
            output_size=Z.shape[1],
            output_transfer='identity',
            hidden_sizes=[hyper.units_per_layer]*1,
            hidden_transfers=["rectifier"]*1
        )
        target_distribution = pm.DiagGaussianNoise(predictor)
        targets = tm.Merge(target_distribution, predictor, tm.Flatten(predictor['parameters'], flat_key="to_be_randomized"))

        params_base = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),)))
        normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)] + [dm.LocScaleTransform()]
        # LocScaleTransform for better working with PlanarTransforms
        params = params_base
        for transform in normflows:
            params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

        prior = pm.Gauss(tm.total_size(targets['to_be_randomized']), init_var=np.exp(-2* hyper.minus_log_s))
        prior = tm.fix_params(prior)
        model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

        _model = model
        _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
        _model = tm.Merge(_model, tm.Flatten(_model['parameters']))


        # # Optimizer

        loss = tm.loss_variational(_model)
        tm.reduce_all_identities()

        n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
        climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

        def weights_regularizer_1epoch():
            for i in range(1, n_batches+1):
                yield 2**(n_batches - i) / (2**n_batches - 1)

        assert len(list(weights_regularizer_1epoch())) == n_batches

        optimizer_kwargs = tm.numericalize(loss, _model['flat'],
            batch_mapreduce=summap,
            annealing_combiner=tm.AnnealingCombiner(
                weights_regularizer=cycle(weights_regularizer_1epoch())
            ),
            adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.01),
        #     profile=True,
        #     mode='FAST_COMPILE',
        )

        opt = optimizer(
            identifier=hyper.opt_identifier,
            step_rate=hyper.opt_step_rate,
            momentum=hyper.opt_momentum,
            decay=hyper.opt_decay,
            offset=hyper.opt_offset,
            args=climin_args,
            **tm.climin_kwargs(optimizer_kwargs)
        )


        # start values:
        hyper.best_val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)

        last_improvement_epoch = 0
        for info in every(n_batches, opt):
            current_epoch = info['n_iter']//n_batches
            if current_epoch - last_improvement_epoch > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)
            if val_loss < hyper.best_val_loss:
                last_improvement_epoch = current_epoch
                hyper.best_parameters = opt.wrt
                hyper.best_val_loss = val_loss
            hyper.val_loss.append(val_loss)

            # visualize training loss for comparison:
            training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            hyper.train_loss.append(training_loss)

        sql_session.commit()  # this updates all set information within sqlite database



        # Model Normalizing flow 2
        # ==========================

        # this is extremely useful to tell everything the default sizes
        input = tm.as_tensor_variable(X[0], name="X")

        predictor = dm.Mlp(
            input=input,
            output_size=Z.shape[1],
            output_transfer='identity',
            hidden_sizes=[hyper.units_per_layer] * 1,
            hidden_transfers=["rectifier"] * 1
        )
        target_distribution = pm.DiagGaussianNoise(predictor)
        targets = tm.Merge(target_distribution, predictor, tm.Flatten(predictor['parameters'], flat_key="to_be_randomized"))

        params_base = pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),))
        normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]  # no LocScaleTransform
        # LocScaleTransform for better working with PlanarTransforms
        params = params_base
        for transform in normflows:
            params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

        prior = pm.Gauss(tm.total_size(targets['to_be_randomized']), init_var=np.exp(-2 * hyper.minus_log_s))
        prior = tm.fix_params(prior)
        model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

        _model = model
        _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
        _model = tm.Merge(_model, tm.Flatten(_model['parameters']))

        # # Optimizer

        loss = tm.loss_variational(_model)
        tm.reduce_all_identities()

        n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
        climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))


        def weights_regularizer_1epoch():
            for i in range(1, n_batches + 1):
                yield 2 ** (n_batches - i) / (2 ** n_batches - 1)


        assert len(list(weights_regularizer_1epoch())) == n_batches

        optimizer_kwargs = tm.numericalize(loss, _model['flat'],
            batch_mapreduce=summap,
            annealing_combiner=tm.AnnealingCombiner(
               weights_regularizer=cycle(weights_regularizer_1epoch())
            ),
            adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.01),
        #     profile=True,
        #     mode='FAST_COMPILE',
        )

        opt = optimizer(
            identifier=hyper.opt_identifier,
            step_rate=hyper.opt_step_rate,
            momentum=hyper.opt_momentum,
            decay=hyper.opt_decay,
            offset=hyper.opt_offset,
            args=climin_args,
            **tm.climin_kwargs(optimizer_kwargs)
        )

        # start values:
        hyper.alternative_best_val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)

        last_improvement_epoch = 0
        for info in every(n_batches, opt):
            current_epoch = info['n_iter'] // n_batches
            if current_epoch - last_improvement_epoch > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)
            if val_loss < hyper.alternative_best_val_loss:
                last_improvement_epoch = current_epoch
                hyper.alternative_best_parameters = opt.wrt
                hyper.alternative_best_val_loss = val_loss
            hyper.alternative_val_loss.append(val_loss)

            # visualize training loss for comparison:
            training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            hyper.alternative_train_loss.append(training_loss)

        sql_session.commit()  # this updates all set information within sqlite database




        # baseline run - WITHOUT normalizing flow
        # =======================================

        # this is extremely useful to tell everything the default sizes
        input = tm.as_tensor_variable(X[0], name="X")

        predictor = dm.Mlp(
            input=input,
            output_size=Z.shape[1],
            output_transfer='identity',
            hidden_sizes=[hyper.units_per_layer] * 1,
            hidden_transfers=["rectifier"] * 1
        )
        target_distribution = pm.DiagGaussianNoise(predictor)
        targets = tm.Merge(target_distribution, predictor,
                           tm.Flatten(predictor['parameters'], flat_key="to_be_randomized"))

        params = pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),))

        prior = pm.Gauss(tm.total_size(targets['to_be_randomized']), init_var=np.exp(-2 * hyper.minus_log_s))
        prior = tm.fix_params(prior)
        model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

        _model = model
        _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
        _model = tm.Merge(_model, tm.Flatten(_model['parameters']))

        # # Optimizer

        loss = tm.loss_variational(_model)
        tm.reduce_all_identities()

        n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
        climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))


        def weights_regularizer_1epoch():
            for i in range(1, n_batches + 1):
                yield 2 ** (n_batches - i) / (2 ** n_batches - 1)


        assert len(list(weights_regularizer_1epoch())) == n_batches

        optimizer_kwargs = tm.numericalize(loss, _model['flat'],
            batch_mapreduce=summap,
            annealing_combiner=tm.AnnealingCombiner(
               weights_regularizer=cycle(weights_regularizer_1epoch())
            ),
            adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.01),
        #     profile=True,
        #     mode='FAST_COMPILE',
        )

        opt = optimizer(
            identifier=hyper.opt_identifier,
            step_rate=hyper.opt_step_rate,
            momentum=hyper.opt_momentum,
            decay=hyper.opt_decay,
            offset=hyper.opt_offset,

            args=climin_args,
            **tm.climin_kwargs(optimizer_kwargs)
        )

        # start values:
        hyper.baseline_best_val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)

        last_improvement_epoch = 0
        for info in every(n_batches, opt):
            current_epoch = info['n_iter'] // n_batches
            if current_epoch - last_improvement_epoch > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)
            if val_loss < hyper.baseline_best_val_loss:
                last_improvement_epoch = current_epoch
                hyper.baseline_best_parameters = opt.wrt
                hyper.baseline_best_val_loss = val_loss
            hyper.baseline_val_loss.append(val_loss)

            # visualize training loss for comparison:
            training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            hyper.baseline_train_loss.append(training_loss)

        sql_session.commit()  # this updates all set information within sqlite database

    except Exception as e:
        with open(os.path.join(__path__, 'hyperparameters2%s_errors.txt' % suffix), "a") as myfile:
            error = """
LAST HYPER: %s
ORIGINAL ERROR: %s
""" % (pformat(hyper.__dict__), traceback.format_exc())
            myfile.write(error)
