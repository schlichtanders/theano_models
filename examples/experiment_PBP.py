# coding: utf-8
# TODO add deterministic normflow versions (i.e. including normflow within deterministic part)!! both for normflow and baseline
from __future__ import division

import contextlib
import os, platform, sys, traceback
from pprint import pformat, pprint
import numpy as np
from climin.util import optimizer
from itertools import repeat, cycle, islice, izip
import random

from schlichtanders.mycontextmanagers import ignored

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
from copy import copy
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

@contextlib.contextmanager
def log_exceptions(title, *exceptions):
    if not exceptions:
        exceptions = Exception
    try:
        yield
    except exceptions:
        with open(os.path.join(__path__, 'experiment_square%s_errors.txt' % suffix), "a") as myfile:
            error = """
%s
------------
LAST HYPER: %s
ORIGINAL ERROR: %s""" % (title, pformat(hyper.__dict__), traceback.format_exc())
            myfile.write(error)


# # Hyperparameters

engine = create_engine('sqlite:///' + os.path.join(__path__, 'experiment_square%s.db' % suffix))
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
    
    # normflows:
    normflows_best_val_loss = Column(Float)
    normflows_best_parameters = Column(PickleType, nullable=True)
    normflows_train_loss = Column(PickleType)
    normflows_val_loss = Column(PickleType)
    normflows_epochs = Column(Integer)

    # normflows2:
    normflows2_best_val_loss = Column(Float)
    normflows2_best_parameters = Column(PickleType, nullable=True)
    normflows2_train_loss = Column(PickleType)
    normflows2_val_loss = Column(PickleType)
    normflows2_epochs = Column(Integer)

    # normflows det:
    normflowsdet_best_val_loss = Column(Float)
    normflowsdet_best_parameters = Column(PickleType, nullable=True)
    normflowsdet_train_loss = Column(PickleType)
    normflowsdet_val_loss = Column(PickleType)
    normflowsdet_epochs = Column(Integer)

    # normflows det2:
    normflowsdet2_best_val_loss = Column(Float)
    normflowsdet2_best_parameters = Column(PickleType, nullable=True)
    normflowsdet2_train_loss = Column(PickleType)
    normflowsdet2_val_loss = Column(PickleType)
    normflowsdet2_epochs = Column(Integer)

    # mixture:
    mixture_best_val_loss = Column(Float)
    mixture_best_parameters = Column(PickleType, nullable=True)
    mixture_train_loss = Column(PickleType)
    mixture_val_loss = Column(PickleType)
    mixture_epochs = Column(Integer)

    # normflows maximum likelihood:
    normflowsml_best_val_loss = Column(Float)
    normflowsml_best_parameters = Column(PickleType, nullable=True)
    normflowsml_train_loss = Column(PickleType)
    normflowsml_val_loss = Column(PickleType)
    normflowsml_epochs = Column(Integer)

    # baseline:
    baseline_best_val_loss = Column(Float)
    baseline_best_parameters = Column(PickleType, nullable=True)
    baseline_train_loss = Column(PickleType)
    baseline_val_loss = Column(PickleType)
    baseline_epochs = Column(Integer)

    def __init__(self, hyper_dict=None):  # we directly refer to dict as sqlalchemy deletes the dict once committed (probably for detecting changes
        if hyper_dict is not None:
            for k, v in hyper_dict.iteritems():
                if not k.startswith("_"):
                    setattr(self, k, copy(v))
            self.init_results()
            return
        self.datasetname = datasetname
        # hyper parameters:
        self.max_epochs_without_improvement = 30
        self.batch_size = random.choice([1,10, 100])
        self.average_n = 1
        self.units_per_layer = 50
        self.minus_log_s = random.choice([1,2,3,4,5,6,7])
        # the prior is learned together with the other models in analogy to the paper Probabilistic Backpropagation
        
        self.n_normflows = random.choice([1,2,3,4,8,20])  #32 is to much for theano... unfortunately
        
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
        for prefix in ['normflows_', 'normflows2_', 'normflowsdet_', 'normflowsdet2_', 'mixture_', 'normflowsml_', 'baseline_']:
            setattr(self, prefix + "best_parameters", None)
            setattr(self, prefix + "best_val_loss", inf)
            setattr(self, prefix + "train_loss", [])
            setattr(self, prefix + "val_loss", [])
            setattr(self, prefix + "epochs", 0)

Base.metadata.create_all()
Session = sessionmaker(bind=engine)
sql_session = Session()


# optimization routine
# ====================
def optimize(prefix, loss, parameters):
    if prefix and not prefix.endswith("_"):  # source of bugs
        prefix += "_"
    tm.reduce_all_identities()

    n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

    def weights_regularizer_1epoch():
        for i in range(1, n_batches+1):
            yield 2**(n_batches - i) / (2**n_batches - 1)

    assert len(list(weights_regularizer_1epoch())) == n_batches

    optimizer_kwargs = tm.numericalize(loss, parameters,
        batch_mapreduce=summap,
        annealing_combiner=tm.AnnealingCombiner(
            weights_regularizer=cycle(weights_regularizer_1epoch())
        ),
        adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.1),
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
    setattr(hyper, prefix + "best_val_loss ",
            optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True))

    last_improvement_epoch = 0
    # val_losses = getattr(hyper, prefix + "val_loss")
    # train_losses = getattr(hyper, prefix + "train_loss")
    for info in every(n_batches, opt):
        current_epoch = info['n_iter']//n_batches
        setattr(hyper, prefix + "epochs", current_epoch)
        if current_epoch - last_improvement_epoch > hyper.max_epochs_without_improvement:
            break
        # collect and visualize validation loss for choosing the best model
        val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, no_annealing=True)
        if val_loss < getattr(hyper, prefix + "best_val_loss"):
            last_improvement_epoch = current_epoch
            setattr(hyper, prefix + "best_parameters", opt.wrt)
            setattr(hyper, prefix + "best_val_loss", val_loss)
        # val_losses.append(val_loss)

        # visualize training loss for comparison:
        # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
        # train_losses.append(training_loss)

    sql_session.commit()  # this updates all set information within sqlite database


def optimizeExp(prefix, loss, parameters):
    if prefix and not prefix.endswith("_"):  # source of bugs
        prefix += "_"
    tm.reduce_all_identities()

    n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

    optimizer_kwargs = tm.numericalizeExp(loss, parameters,
        adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.1),
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
    setattr(hyper, prefix + "best_val_loss ",
            optimizer_kwargs['num_loss'](opt.wrt, VZ, VX))

    last_improvement_epoch = 0
    # val_losses = getattr(hyper, prefix + "val_loss")
    # train_losses = getattr(hyper, prefix + "train_loss")
    for info in every(n_batches, opt):
        current_epoch = info['n_iter']//n_batches
        setattr(hyper, prefix + "epochs", current_epoch)
        if current_epoch - last_improvement_epoch > hyper.max_epochs_without_improvement:
            break
        # collect and visualize validation loss for choosing the best model
        val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX)
        if val_loss < getattr(hyper, prefix + "best_val_loss"):
            last_improvement_epoch = current_epoch
            setattr(hyper, prefix + "best_parameters", opt.wrt)
            setattr(hyper, prefix + "best_val_loss", val_loss)
        # val_losses.append(val_loss)

        # visualize training loss for comparison:
        # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
        # train_losses.append(training_loss)

    sql_session.commit()  # this updates all set information within sqlite database


# Main Loop
# =========

while True:
    for _i in range(3):  # repeat, taking slightly different starting parameters each time
        if _i == 0:
            hyper = RandomHyper()
            hyper_dict = copy(hyper.__dict__)
            pprint(hyper_dict)
            sql_session.add(hyper)
        else:
            hyper = RandomHyper(hyper_dict)
            sql_session.add(hyper)

        # reset hard the saved models:
        dm.InvertibleModel.INVERTIBLE_MODELS = []
        tm.Model.all_models = []

        # baseline run
        # ============

        with log_exceptions("baseline"):
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

            params = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))

            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))
            loss = tm.loss_variational(_model)

            optimize("baseline", loss, _model['flat'])

        # Model Normalizing flow
        # ======================
        with log_exceptions("normflows"):
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

            params_base = pm.StandardGaussian(output_shape=(tm.total_size(targets['to_be_randomized']),))
            normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)] + [dm.LocScaleTransform(independent_scale=True)]
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2* hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))

            loss = tm.loss_variational(_model)

            optimize("normflows", loss, _model['flat'])


        # Model Normalizing flow 2
        # ==========================
        with log_exceptions("normflows2"):
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

            params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
            normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]  # no LocScaleTransform
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))
            loss = tm.loss_variational(_model)

            optimize("normflows2", loss, _model['flat'])


        # Model Normalizing flow Deterministic
        # ====================================
        with log_exceptions("normflowsdet"):
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
            target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized")
            for _ in range(hyper.n_normflows - 1):
                target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)
            target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

            targets = tm.Merge(target_distribution, predictor,
                               tm.Flatten(predictor['parameters']))
            total_size = tm.total_size(targets['flat'])
            targets['flat'] = target_normflow
            targets = tm.Merge(targets, target_normflow)

            params = pm.StandardGaussian(output_shape=(total_size,))
            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model,
                              tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))

            loss = tm.loss_variational(_model)

            optimize("normflowsdet", loss, _model['flat'])

        # Model Normalizing flow 2
        # ==========================
        with log_exceptions("normflowsdet2"):
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

            target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized")
            for _ in range(hyper.n_normflows - 1):
                target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)

            targets = tm.Merge(target_distribution, predictor,
                               tm.Flatten(predictor['parameters']))
            total_size = tm.total_size(targets['flat'])
            targets['flat'] = target_normflow
            targets = tm.Merge(targets, target_normflow)

            params = pm.DiagGauss(output_size=total_size)
            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model,
                              tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))
            loss = tm.loss_variational(_model)

            optimize("normflowsdet2", loss, _model['flat'])


        # Mixture
        # =======
        with log_exceptions("mixture"):
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

            # the number of parameters comparing normflows and mixture of gaussians match perfectly (the only exception is
            # that we spend an additional parameter when modelling n psumto1 with n parameters instead of (n-1) within softmax
            total_size = tm.total_size(targets['to_be_randomized'])
            mixture_comps = [pm.DiagGauss(output_size=total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
            params = pm.Mixture(*mixture_comps)
            prior = tm.fix_params(pm.Gauss(output_shape=(tm.total_size(targets['to_be_randomized']),),
                                           init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)

            _model = model
            _model = tm.Merge(_model,
                              tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv),
                              tm.Reparameterize(_model['parameters_psumto1'], tm.softmax, tm.softmax_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))
            loss = tm.loss_variational(_model)

            optimize("mixture", loss, _model['flat'])


        # Normflows Maximum Likelihood
        # ============================
        with log_exceptions("normflowsml"):
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

            params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
            normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]  # no LocScaleTransform
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            targets['to_be_randomized'] = params
            model = tm.Merge(targets, params)

            _model = model
            # _model = tm.Merge(_model, tm.Reparameterize(_model['parameters_positive'], tm.softplus, tm.softplus_inv))
            _model = tm.Merge(_model,
                              tm.Reparameterize(_model['parameters_positive'], tm.squareplus, tm.squareplus_inv))
            _model = tm.Merge(_model, tm.Flatten(_model['parameters']))
            loss = tm.loss_probabilistic(_model)

            optimizeExp("normflowsml", loss, _model['flat'])