#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import signal
import climin
import climin.util
import climin.mathadapt as ma
from schlichtanders.myobjects import Namespace

__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'


CTRL_C_FLAG = False


def run_optimizer(optimizer, stop, report, extra_logging=None):
    """
    Denote the iterative output with ``info`` in the following
    :param optimizer: supports iterative interface
    :param stop: if ``stop(info)=True`` optimization stops
    :param report: if ``report(info)=True`` yields ``info``
    :param extra_logging:

        Gets called on every report like ``extra_logging(optimizer, info)`` before ``info`` is yielded.
        Hence, it can alter ``info``, e.g. add new informations
    :return: adapted optimizer generator
    """
    if isinstance(extra_logging, dict):
        extra_logging = dict.items()

    global CTRL_C_FLAG
    CTRL_C_FLAG = False
    signal.signal(signal.SIGINT, self._ctrl_c_handler)

    for info in optimizer:
        if report(info) or stop(info) or CTRL_C_FLAG:
            for func in extra_logging:
                func(optimizer, info)

            yield info
            if stop(info) or CTRL_C_FLAG:
                break


def extra_logging_scores(score, val_data, train_data=None):
    ns = Namespace()
    ns.best_loss = float('inf')
    ns.best_pars = None  # shouldn't be needed

    def extra_logging(optimizer, info):
        if 'loss' not in info and train_data is not None:
            info['loss'] = ma.scalar(score(*train_data))

        info['val_loss'] = ma.scalar(score(*val_data))

        if info['val_loss'] < ns.best_loss:
            ns.best_loss = info['val_loss']
            ns.best_pars = optimizer.wrt.copy()

        info['best_loss'] = ns.best_loss
        info['best_pars'] = ns.best_pars

    return extra_logging


def _ctrl_c_handler(self, signal, frame):
    global CTRL_C_FLAG
    CTRL_C_FLAG = True