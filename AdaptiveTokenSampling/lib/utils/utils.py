from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import shutil
import tensorwatch as tw
from lib.utils.comm import comm
from ptflops import get_model_complexity_info
from fvcore.nn.flop_count import flop_count

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def summary_model_on_master(model, config, output_dir, copy):

        if True:
            try:
                logging.info("== model_stats by tensorwatch ==")
                df = tw.model_stats(
                    model,
                    (1, 3, config.input_size, config.input_size),
                )
                df.to_html(os.path.join(output_dir, "model_summary.html"))
                df.to_csv(os.path.join(output_dir, "model_summary.csv"))
                msg = "*" * 20 + " Model summary " + "*" * 20
                logging.info(
                    "\n{msg}\n{summary}\n{msg}".format(msg=msg, summary=df.iloc[-1])
                )
                logging.info("== model_stats by tensorwatch ==")
            except Exception:
                logging.error("=> error when run model_stats")

            try:
                logging.info("== get_model_complexity_info by ptflops ==")
                macs, params = get_model_complexity_info(
                    model,
                    (3, config.input_size, config.input_size),
                    as_strings=True,
                    print_per_layer_stat=True,
                    verbose=True,
                )
                logging.info(f"=> FLOPs: {macs:<8}, params: {params:<8}")
                logging.info("== get_model_complexity_info by ptflops ==")
                # pdb.set_trace()
            except Exception:
                logging.error("=> error when run get_model_complexity_info")

def calculate_flops(model, input):
    flops, _ = flop_count(model, input)
    return sum(flops.values())