import argparse
import json
from utils_dip import utils


def transfer_bool(s:str):
    if s is "False":
        return False
    elif s is "True":
        return True
    return None


def parse_args(config_file='configs.json'):
    all_config = json.load(open(config_file))

    name = all_config["name"]
    double_parameterization = transfer_bool(all_config["double parameterization"])

    data_config = all_config["data"]
    data_path = data_config["data path"]
    sparse_noise = transfer_bool(data_config["noisy"])
    if "super resolution" in name:
        factor = data_config["factor"]
    else:
        factor = None

    trainer_config = all_config["trainer"]
    num_iter = trainer_config["number iter"]
    optimizer1 = trainer_config["optimizer 1"]
    if trainer_config["optimizer 2"] is "None":
        optimizer2 = None
    else:
        optimizer2 = trainer_config["optimizer 2"]

    l1 = trainer_config["l1"]
    if trainer_config["l2"] is "None":
        l2 = None
    else:
        l2 = trainer_config["l2"]
    snapshot = trainer_config["snapshot"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default=name,
                        help='the name of experiment')
    parser.add_argument("--over parameterization", type=bool, default=double_parameterization,
                        help='whether to use the method of double over parameterization')
    parser.add_argument("--noise", type=bool, default=sparse_noise,
                        help='whether to add sparse noise')
    parser.add_argument("--factor", type=int, default=factor,
                        help='Super resolution factor scale')
    parser.add_argument("--num iter", type=int, default=num_iter,
                        help='number of iterative weight updates, DEFAULT=' + str(num_iter))
    parser.add_argument("--optimizer 1", type=str, default=optimizer1,
                        help='choose the first optimizer for main DIP model')
    parser.add_argument("--optimizer 2", type=str, default=optimizer2,
                        help='choose the second optimizer for Over Parameterization Part')
    parser.add_argument("--l1", type=float, default=l1,
                        help='learning rate for optimizer 1')
    parser.add_argument("--l2", type=float, default=l2,
                        help='learning rate for optimizer 2')
    parser.add_argument('--snapshot', type=int, default=snapshot,
                        help='steps to record image')
    
    args = parser.parse_args()

    # set values for data-specific configurations

    return args
