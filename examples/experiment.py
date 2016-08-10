# coding: utf-8
from __future__ import division

import contextlib
import os, platform, sys, traceback
from functools import partial
from pprint import pformat, pprint
import numpy as np
from theano.gof import MethodNotDefined

from climin.util import optimizer
from itertools import repeat, cycle, islice, izip
import random

from schlichtanders.mycontextmanagers import ignored

inf = float("inf")

from schlichtanders.myfunctools import compose, meanmap, summap, compose_fmap, Average
from schlichtanders.mygenerators import eatN, chunk, chunk_list, every, takeN
from schlichtanders.myobjects import NestedNamespace, Namespace

import theano_models as tm
import theano_models.deterministic_models as dm
import theano_models.probabilistic_models as pm
from theano_models import data

from sklearn import cross_validation
from theano.tensor.shared_randomstreams import RandomStreams
import theano

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from copy import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

tm.inputting_references.update(['to_be_randomized'])
tm.inputting_references, tm.outputting_references

EPS = 1e-4

pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(5e8)), RandomStreams())

__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

# defaults:
foldername = "experiment"
filename = "several"
datasetname = "boston"

#overwrite as far as given:
if sys.argv > 3:
    foldername, filename, datasetname = sys.argv[1:4]
elif sys.argv > 2:
    filename, datasetname = sys.argv[1:3]
elif sys.argv > 1:
    datasetname = sys.argv[1]


class Track(object):
    def __getattr__(self, item):
        return tm.track_model(getattr(tm, item))
track = Track()

# # Data
#     # datasetnames = ["boston", "concrete", "energy", "kin8nm", "naval", "powerplant", "protein", "winered", "yacht", "year"]
#     datasetnames = ["boston", "concrete", "energy", "kin8nm", "powerplant", "winered", "yacht"]

# "naval" is not possible as standardization leads to a nan column. Hence we do not know which to use
# datasetname = "concrete"


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

# TODO the above randomly splits the dataset, which should be averaged out ideally... however with same initial parameters... that makes it difficult

@contextlib.contextmanager
def log_exceptions(title, *exceptions):
    if not exceptions:
        exceptions = Exception
    try:
        yield
    except exceptions:
        with open(os.path.join(__path__, foldername, '%s_errors.txt' % filename), "a") as myfile:
            error = """
%s
------------
LAST HYPER: %s
ORIGINAL ERROR: %s""" % (title, pformat(hyper.__dict__), traceback.format_exc())
            myfile.write(error)


def RMSE(PX, Z):
    return np.sqrt(((PX - Z) ** 2).mean())

def nRMSE(PX, Z):
    return RMSE(PX*Z_std + Z_mean, Z*Z_std + Z_mean)

# # Hyperparameters
with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))
engine = create_engine('sqlite:///' + os.path.join(__path__, foldername, '%s.db' % filename))
Base = declarative_base(bind=engine)


class RandomHyper(Base):
    __tablename__ = "hyper"
    id = Column(Integer, primary_key=True)

    # hyper parameters:
    datasetname = Column(String)
    max_epochs_without_improvement = Column(Integer)
    logP_average_n = Column(Integer)
    errorrate_average_n = Column(Integer)
    exp_average_n = Column(Integer)
    units_per_layer = Column(Integer)
    n_layers = Column(Integer)
    minus_log_s = Column(Integer)
    batch_size = Column(Integer)
    
    n_normflows = Column(Integer)
    
    opt_identifier = Column(String)
    opt_momentum = Column(Float)
    opt_offset = Column(Float)
    opt_decay = Column(Float)
    opt_step_rate = Column(Float)

    for _prefix in ['baselinedet_', 'baselinedetplus_',
                    'baseline_', 'baselineplus_',
                    'mixture_', 'mixtureml_',
                    'planarflow_', 'planarflowdet_', 'planarflowml_',
                    'radialflow_', 'radialflowdet_', 'radialflowml_']:
        exec("""
{0}best_val_loss = Column(Float)
{0}best_parameters = Column(PickleType, nullable=True)
{0}train_loss = Column(PickleType)
{0}val_loss = Column(PickleType)
{0}epochs = Column(Integer)
{0}init_params = Column(PickleType, nullable=True)
{0}val_error_rate = Column(Float)""".format(_prefix))

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
        # batch_size=2 for comparison with maximum-likelihood (dimensions error was thrown in exactly those cases for batch_size=1
        # there are still erros with batch_size=2 for some weird reasons... don't know. I hope this is not crucial.
        self.batch_size = random.choice([1, 10, 50, 100])
        self.logP_average_n = 1  # TODO random.choice([1,10])
        self.errorrate_average_n = 20
        self.exp_average_n = 20
        self.exp_ratio_estimator = random.choice([None, "grouping", "firstorder"])
        self.n_layers = 1  # random.choice([1, 2])  # for comparison of parameters, n_layers=1 has crucial advantages
        self.units_per_layer = 50  # random.choice([50, 200]) - check whether baselineplus has an effect
        self.minus_log_s = random.choice([1,2,3,4,5,6,7])
        # the prior is learned together with the other models in analogy to the paper Probabilistic Backpropagation
        
        self.n_normflows = random.choice([1,2,3,4,8,20])  #32 is to much for theano... unfortunately
        
        self.opt_identifier = random.choice(["adadelta", "adam", "rmsprop"])
        if self.opt_identifier == "adadelta":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.9, 1)])
            # self.opt_offset = random.choice([5e-5, 1e-8])
        elif self.opt_identifier == "adam":
            self.opt_momentum = random.choice([np.random.uniform(0, 0.01), np.random.uniform(0.8, 0.93)])
            # self.opt_offset = 10 ** -np.random.uniform(3, 4)
        elif self.opt_identifier == "rmsprop":
            self.opt_momentum = random.choice([np.random.uniform(0.002, 0.008), np.random.uniform(0.9, 1)])
            # self.opt_offset = np.random.uniform(0, 0.000045)
        # self.opt_step_rate = random.choice([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        self.opt_step_rate = random.choice([1e-3, 1e-4, 1e-5, 1e-6])
        self.opt_decay = np.random.uniform(0.78, 1)
        
        self.init_results()
    
    def init_results(self):
        # extra for being able to reset results for loaded hyperparameters
        for prefix in ['baselinedet_', 'baselinedetplus_',
                       'baseline_', 'baselineplus_',
                       'mixture_', 'mixtureml_',
                       'planarflow_', 'planarflowdet_', 'planarflowml_',
                       'radialflow_', 'radialflowdet_', 'radialflowml_']:
            setattr(self, prefix + "best_parameters", None)
            setattr(self, prefix + "best_val_loss", inf)
            setattr(self, prefix + "train_loss", [])
            setattr(self, prefix + "val_loss", [])
            setattr(self, prefix + "best_epoch", 0)
            setattr(self, prefix + "init_params", None)
            setattr(self, prefix + "val_error_rate", inf)

Base.metadata.create_all()
Session = sessionmaker(bind=engine)
sql_session = Session()


# optimization routine
# ====================
def optimize(prefix, model, loss, parameters, type='annealing'):
    print prefix
    if prefix and not prefix.endswith("_"):  # source of bugs
        prefix += "_"
    tm.reduce_all_identities()

    n_batches = X.shape[0] // hyper.batch_size  # after this many steps we went through the whole data set once
    climin_args = izip(izip(chunk(hyper.batch_size, cycle(Z)), chunk(hyper.batch_size, cycle(X))), repeat({}))

    if type == "ml": # maximum likelihood
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
        )
    elif type == "ml_exp_average":
        numericalize_kwargs = dict(
            batch_mapreduce=meanmap,
            exp_average_n=hyper.exp_average_n,
            exp_ratio_estimator=hyper.exp_ratio_estimator,
        )
    elif type == "annealing":
        def weights_regularizer_1epoch():
            for i in range(1, n_batches + 1):
                yield 2 ** (n_batches - i) / (2 ** n_batches - 1)
        assert len(list(weights_regularizer_1epoch())) == n_batches
        numericalize_kwargs = dict(
            batch_mapreduce=summap,  # meaning is/must be done in Annealing
            annealing_combiner=tm.AnnealingCombiner(
                weights_regularizer=cycle(weights_regularizer_1epoch())
            ),
        )
    else:
        raise ValueError("Unkown type %s" % type)

    def _optimize(mode='FAST_RUN'):
        optimizer_kwargs = tm.numericalize(
            loss, parameters,
            adapt_init_params=lambda ps: ps + np.random.normal(size=ps.size, scale=0.5), # better more initial randomness
            # profile=True,
            mode=mode, #'FAST_COMPILE' if hyper.n_normflows > 10 else 'FAST_RUN', # error that theano cannot handle ufuncs with more than 32 arguments
            **numericalize_kwargs
        )

        opt = optimizer(
            identifier=hyper.opt_identifier,
            step_rate=hyper.opt_step_rate,
            momentum=hyper.opt_momentum,
            decay=hyper.opt_decay,
            # offset=hyper.opt_offset,
            args=climin_args,
            **tm.climin_kwargs(optimizer_kwargs)
        )

        val_kwargs = {}
        if type == "annealing":
            val_kwargs['no_annealing'] = True
        # start values:
        setattr(hyper, prefix + "init_params", copy(opt.wrt))
        setattr(hyper, prefix + "best_val_loss",
                optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
                )
                # TODO include this:
                # if hyper.logP_average_n <= 1 or type.startswith("ml") else  # maximum_likelihood already averages over each single data point
                # Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, VZ, VX, **val_kwargs))

        setattr(hyper, prefix + "val_loss", [])
        val_losses = getattr(hyper, prefix + "val_loss")
        # train_losses = getattr(hyper, prefix + "train_loss")
        for info in every(n_batches, opt):
            current_epoch = info['n_iter']//n_batches
            # print current_epoch,
            if current_epoch - getattr(hyper, prefix + "best_epoch") > hyper.max_epochs_without_improvement:
                break
            # collect and visualize validation loss for choosing the best model
            val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            # TODO include this
            # if hyper.logP_average_n <= 1 or type.startswith("ml"):  # maximum_likelihood already averages over each single data point
            #     val_loss = optimizer_kwargs['num_loss'](opt.wrt, VZ, VX, **val_kwargs)
            # else:  # as we use batch_common_rng = True by default, for better comparison, average over several noisy weights:
            #     val_loss = Average(hyper.logP_average_n)(optimizer_kwargs['num_loss'], opt.wrt, VZ, VX, **val_kwargs)
            if val_loss < getattr(hyper, prefix + "best_val_loss") - EPS:
                setattr(hyper, prefix + "best_epoch", current_epoch)
                setattr(hyper, prefix + "best_parameters", copy(opt.wrt))  # copy is needed as climin works inplace on array
                setattr(hyper, prefix + "best_val_loss", val_loss)
            val_losses.append(val_loss)

            # visualize training loss for comparison:
            # training_loss = optimizer_kwargs['num_loss'](opt.wrt, Z[:10], X[:10], no_annealing=True)
            # train_losses.append(training_loss)
        print current_epoch

    try:
        _optimize()
    except MethodNotDefined:  # this always refers to the limit of 32 nodes per ufunc... weird issue
        _optimize(mode='FAST_COMPILE')

    if getattr(hyper, prefix + "best_parameters") is not None:  # sometimes the above does not even run one epoch
        # test error rate:
        sampler = theano.function([parameters] + model['inputs'], model['outputs'], allow_input_downcast=True)
        PVX = np.array(
            [Average(hyper.errorrate_average_n)(sampler, getattr(hyper, prefix + "best_parameters"), x) for x in VX])
        setattr(hyper, prefix + 'val_error_rate', nRMSE(PVX, VZ))

    sql_session.commit()  # this updates all set information within sqlite database


# Main Loop
# =========

# capital, as these construct models
Reparam = tm.as_proxmodel('parameters')(tm.prox_reparameterize)
Flat = tm.as_proxmodel("to_be_randomized")(tm.prox_flatten)

while True:
    for _i in range(3):  # repeat, taking slightly different starting parameters each time
        if _i == 0:
            hyper = RandomHyper()
            hyper_dict = copy(hyper.__dict__)
            pprint(hyper_dict)
            sql_session.add(hyper)
        else:
            hyper = RandomHyper(hyper_dict) # new hyper with same parameters
            sql_session.add(hyper)

        # reset hard the saved models:
        dm.InvertibleModel.INVERTIBLE_MODELS = []
        tm.Model.all_models = []

        # baselinedet
        # ===========

        with log_exceptions("baselinedet"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            model = tm.Merge(target_distribution, predictor)

            loss = tm.loss_probabilistic(model)  # TODO no regularizer yet ...

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("baselinedet", model, loss, flat, type="ml")


        # baselinedet plus
        # ================

        with log_exceptions("baselinedetplus"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[(hyper.units_per_layer+ hyper.units_per_layer*(hyper.n_normflows*2))] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            model = tm.Merge(target_distribution, predictor)

            loss = tm.loss_probabilistic(model)  # TODO no regularizer yet ...

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("baselinedetplus", model, loss, flat, type="ml")

        # baseline
        # ========

        with log_exceptions("baseline"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            _total_size = tm.total_size(targets['to_be_randomized'])
            params = pm.DiagGauss(output_size=_total_size)
            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("baseline", model, loss, flat)

        # baseline plus
        # =============

        with log_exceptions("baselineplus"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[(hyper.units_per_layer + hyper.units_per_layer*(hyper.n_normflows*2))] * hyper.n_layers,  # TODO this formula does not work for hyper.n_layers >= 2, too many parameters
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            _total_size = tm.total_size(targets['to_be_randomized'])
            params = pm.DiagGauss(output_size=_total_size)
            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("baselineplus", model, loss, flat)


        # planarflow
        # ==========
        with log_exceptions("planarflow"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer]*hyper.n_layers,
                hidden_transfers=["rectifier"]*hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            _total_size = tm.total_size(targets['to_be_randomized'])
            params_base = pm.DiagGauss(output_size=_total_size)
            normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2* hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("planarflow", model, loss, flat)


        # planarflow Deterministic
        # ========================

        with log_exceptions("planarflowdet"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            target_normflow = tm.Merge(dm.PlanarTransform(), inputs="to_be_randomized") # rename inputs is crucial!!
            for _ in range(hyper.n_normflows - 1):
                target_normflow = tm.Merge(dm.PlanarTransform(target_normflow), target_normflow)
            # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))
            _total_size = tm.total_size(targets['to_be_randomized'])
            targets['to_be_randomized'] = target_normflow
            targets = tm.Merge(targets, target_normflow)

            params = pm.DiagGauss(output_size=_total_size)
            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("planarflowdet", model, loss, flat)


        # planarflow Maximum Likelihood
        # =============================
        with log_exceptions("planarflowml"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
            normflows = [dm.PlanarTransform() for _ in range(hyper.n_normflows)]  # no LocScaleTransform
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            targets['to_be_randomized'] = params
            model = tm.Merge(targets, params)
            loss = tm.loss_probabilistic(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("planarflowml", model, loss, flat, type="ml_exp_average")


        # radialflow
        # ==========
        with log_exceptions("radialflow"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            _total_size = tm.total_size(targets['to_be_randomized'])
            params_base = pm.DiagGauss(output_size=_total_size)
            normflows = [dm.RadialTransform() for _ in range(hyper.n_normflows*2)] # *2 as radial flow needs only half of the parameters
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("radialflow", model, loss, flat)


        # radialflow Deterministic
        # ========================

        with log_exceptions("radialflowdet"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            target_normflow = tm.Merge(dm.PlanarTransform(),
                                       inputs="to_be_randomized")  # rename inputs is crucial!!
            for _ in range(hyper.n_normflows*2 - 1): # *2 as radial flow needs only half of the parameters
                target_normflow = tm.Merge(dm.RadialTransform(target_normflow), target_normflow)
            # target_normflow = tm.Merge(dm.LocScaleTransform(target_normflow, independent_scale=True), target_normflow)

            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))
            _total_size = tm.total_size(targets['to_be_randomized'])
            targets['to_be_randomized'] = target_normflow
            targets = tm.Merge(targets, target_normflow)

            params = pm.DiagGauss(output_size=_total_size)
            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("radialflowdet", model, loss, flat)


        # radialflow Maximum Likelihood
        # =============================
        with log_exceptions("radialflowml"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            params_base = pm.DiagGauss(output_size=tm.total_size(targets['to_be_randomized']))
            normflows = [dm.RadialTransform() for _ in range(hyper.n_normflows*2)]  # *2 as radial flow needs only half of the parameters
            # LocScaleTransform for better working with PlanarTransforms
            params = params_base
            for transform in normflows:
                params = tm.normalizing_flow(transform, params)  # returns transform, however with adapted logP

            targets['to_be_randomized'] = params
            model = tm.Merge(targets, params)
            loss = tm.loss_probabilistic(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus,
                                                track.squareplus_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("radialflowml", model, loss, flat, type="ml_exp_average")


        # Mixture
        # =======
        with log_exceptions("mixture"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            # the number of parameters comparing normflows and mixture of gaussians match perfectly (the only exception is
            # that we spend an additional parameter when modelling n psumto1 with n parameters instead of (n-1) within softmax
            _total_size = tm.total_size(targets['to_be_randomized'])
            mixture_comps = [pm.DiagGauss(output_size=_total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
            params = pm.Mixture(*mixture_comps)
            prior = tm.fix_params(pm.Gauss(output_shape=(_total_size,), init_var=np.exp(-2 * hyper.minus_log_s)))
            model = tm.variational_bayes(targets, 'to_be_randomized', params, priors=prior)
            loss = tm.loss_variational(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
            all_params += tm.prox_reparameterize(model['parameters_psumto1'], tm.softmax, tm.softmax_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("mixture", model, loss, flat)


        # Mixture Maximum Likelihood
        # ==========================
        with log_exceptions("mixtureml"):
            # this is extremely useful to tell everything the default sizes
            input = tm.as_tensor_variable(X[0], name="X")

            predictor = dm.Mlp(
                input=input,
                output_size=Z.shape[1],
                output_transfer='identity',
                hidden_sizes=[hyper.units_per_layer] * hyper.n_layers,
                hidden_transfers=["rectifier"] * hyper.n_layers
            )
            target_distribution = pm.DiagGaussianNoise(predictor)
            targets = tm.Merge(target_distribution, predictor, Flat(predictor['parameters']))

            mixture_comps = [pm.DiagGauss(output_size=_total_size) for _ in range(hyper.n_normflows + 1)]  # +1 for base_model
            params = pm.Mixture(*mixture_comps)

            targets['to_be_randomized'] = params
            model = tm.Merge(targets, params)
            loss = tm.loss_probabilistic(model)

            # all_params = tm.prox_reparameterize(model['parameters_positive'], tm.softplus, tm.softplus_inv)
            all_params = tm.prox_reparameterize(model['parameters_positive'], track.squareplus, track.squareplus_inv)
            all_params += tm.prox_reparameterize(model['parameters_psumto1'], tm.softmax, tm.softmax_inv)
            all_params += model['parameters']
            flat = tm.prox_flatten(tm.prox_center(all_params))
            optimize("mixtureml", model, loss, flat, type="ml_exp_average")
