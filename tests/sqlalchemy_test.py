#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from sqlalchemy import Column, Integer, Unicode, UnicodeText, String, PickleType, Float, Boolean
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, make_transient
from sqlalchemy.ext.declarative import declarative_base
import os, platform
import random
import numpy as np
from copy import copy
from pprint import pprint as print
inf = float("inf")

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'
__file__ = os.path.realpath(__file__)
if platform.system() == "Windows":
    from schlichtanders.myos import replace_unc
    __file__ = replace_unc(__file__)
__path__ = os.path.dirname(__file__)
__parent__ = os.path.dirname(__path__)


engine = create_engine('sqlite:///' + os.path.join(__path__, 'test_sql.db'))
Base = declarative_base(bind=engine)

datasetname = "boston"

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

    # mixture:
    mixture_best_val_loss = Column(Float)
    mixture_best_parameters = Column(PickleType, nullable=True)
    mixture_train_loss = Column(PickleType)
    mixture_val_loss = Column(PickleType)
    mixture_epochs = Column(Integer)

    # baseline:
    baseline_best_val_loss = Column(Float)
    baseline_best_parameters = Column(PickleType, nullable=True)
    baseline_train_loss = Column(PickleType)
    baseline_val_loss = Column(PickleType)
    baseline_epochs = Column(Integer)

    def __init__(self, other_hyper_dict=None):  # we directly refer to dict as sqlalchemy deletes the dict once committed (probably for detecting changes
        if other_hyper_dict is not None:
            print("yeah")
            for k, v in other_hyper_dict.iteritems():
                print("k outer = %s" % k)
                if not k.startswith("_"):
                    print("k inner = %s" % k)
                    setattr(self, k, copy(v))
            self.init_results()
            return
        # else:
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
        for prefix in ['normflows_', 'normflows2_', 'mixture_', 'baseline_']:
            setattr(self, prefix + "best_parameters", None)
            setattr(self, prefix + "best_val_loss", inf)
            setattr(self, prefix + "train_loss", [])
            setattr(self, prefix + "val_loss", [])
            setattr(self, prefix + "epochs", 0)

Base.metadata.create_all()
Session = sessionmaker(bind=engine)
sql_session = Session()

hyper = RandomHyper()
d = copy(hyper.__dict__)
print(hyper.__dict__)
sql_session.add(hyper)
hyper.baseline_best_val_loss = 99
sql_session.commit()
print(hyper.__dict__)
print(d)

hyper2 = RandomHyper(d)
sql_session.add(hyper2)
hyper2.mixture_best_val_loss = 78
sql_session.commit()

hyper3 = RandomHyper()
sql_session.add(hyper3)
hyper3.normflows_best_val_loss = 56
sql_session.commit()



