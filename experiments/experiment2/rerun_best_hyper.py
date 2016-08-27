from __future__ import division
from ast import literal_eval

from sklearn import cross_validation
from pprint import pprint

import os, platform, sys
import warnings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import experiment_toy_models
import experiment_util
from schlichtanders.mycontextmanagers import ignored
from schlichtanders.myobjects import NestedNamespace
import evaluation_util as eva

import numpy as np
import theano
import theano_models as tm
from theano.tensor.shared_randomstreams import RandomStreams
import theano_models.probabilistic_models as pm


__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)

sys.path.append(__parent__)

folders_parameters = [["experiment2", "windows_newannealing"]]
folders_parameters = [os.path.join(__parent__, *fp) for fp in folders_parameters]

warnings.filterwarnings("ignore", category=DeprecationWarning)
inf = float('inf')


foldername = "best_test_sampled"
filename = "several"
datasetname = "energy"

#overwrite as far as given:
if len(sys.argv) > 3:
    foldername, filename, datasetname = sys.argv[1:4]
elif len(sys.argv) > 2:
    foldername, filename = sys.argv[1:3]
elif len(sys.argv) > 1:
    foldername = sys.argv[1]

with ignored(OSError):
    os.mkdir(os.path.join(__path__, foldername))

filepath_tests = os.path.join(__path__, foldername, "%s.db" % filename)
# filepath_samples = os.path.join(__path__, foldername, "%s_samples.pkl" % datasetname)
# -------------------------------------------

Hyper = experiment_util.get_hyper()

# # Collect models and find best ones
# best_hyper = eva.get_best_hyper(["toy_windows", "toy_linux"], Hyper, model_prefixes, test_suffix=["best_val_loss"])
def data_gen(hyper):
    return experiment_util.load_and_preprocess_data(hyper.datasetname)


# compute everything:
print "datasetname", datasetname

n_normflows = [2, 5, 10, 25] if "toy" in datasetname else [10, 25]
best_hypers = eva.get_best_hyper_autofix(
    datasetname, folders_parameters,
    test_attrs=["best_val_error"],
    n_normflows=n_normflows,
    modelnames=("baseline", "baselinedet", "planarflow", "planarflowdet", "radialflow", "radialflowdet"))

print "---------------------------------------------------------"
pprint([(h.modelname, h.n_normflows, h.percent, h.best_val_loss, h.best_val_error) for h in best_hypers])  # To see validation performance and whether it makes sense to sample these
print "---------------------------------------------------------"




engine = create_engine('sqlite:///' + filepath_tests)  # os.path.join(__path__, foldername, '%s.db' % filename)
Hyper.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

pm.RNG = NestedNamespace(tm.PooledRandomStreams(pool_size=int(1e8)), RandomStreams())
for h in best_hypers:
    sql_session = Session()
    extra_dict = {k: v for k, v in h.__dict__.iteritems()
                  if k[:4] not in ["best", "mixt", "radi", "plan", "base"] and k not in ["val_loss", "train_loss"]}
    with experiment_util.log_exceptions(filepath_tests + ".errors.txt", h.modelname, extra_dict):
        print("modelname=%s, nn=%i, percent=%g, best_val_loss=%g, best_val_error=%g"
              % (h.modelname, h.n_normflows, h.percent, h.best_val_loss, h.best_val_error))
        new_h = eva.rerun_hyper(h, data_gen)
        sql_session.add(new_h)
        sql_session.commit()

# Sample the approximate posterior distribution for evaluation
# best_hyper_samples = eva.sample_best_hyper(best_hyper_tests, filepath=filepath_samples)
