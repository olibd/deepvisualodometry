import sys
from time import sleep

from comet_ml import Experiment, ExistingExperiment
from comet_ml import Optimizer as CometOptimizer
from git import Repo
import torch
from torch import multiprocessing
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import Loaders
from Common.Helpers import TracebackSignalTimeout
from Common.Helpers import cuda_is_available
from Loaders import load_dataset_dataloaders, load_model, load_optimizer, load_loss, \
    load_all_test_trajectory_dataloaders
from Logging import CometLogger
from ModelTester import ModelTester
from ModelTrainer import ModelTrainer
from Models.Losses import AbstractLoss
from Parameters import Parameters


def launch_training(model: nn.Module, train_dataloader: DataLoader, valid_dataloader, optimizer: Optimizer,
                    loss: AbstractLoss, param: Parameters):
    trainer = ModelTrainer(model, train_dataloader, valid_dataloader, optimizer, loss,
                           param.early_stopping_patience, param.model_backup_destination, param.resume,
                           param.gradient_clipping_value)
    trainer.run(param.epochs)


def launch_testing(model: nn.Module, param: Parameters):
    test_trajectory_dataloaders = load_all_test_trajectory_dataloaders(param)
    tester = ModelTester(model=model, trajectory_dataloaders=test_trajectory_dataloaders,
                         sliding_window_size=param.sliding_window_size,
                         sliding_window_overlap=param.sliding_window_overlap,
                         model_name=param.model)
    tester.run()


def launch_single_GPU_experiment(experiment: Experiment, repo: Repo, param: Parameters):
    setup_comet_experiment(experiment, param, repo)
    loss, model, optimizer, train_dataloader, valid_dataloader = load_experiment_assets(param)
    if param.train:
        print("~~ Launching the training ~~")
        launch_training(model, train_dataloader, valid_dataloader, optimizer, loss, param)
    if param.test:
        print("~~ Testing the model ~~")
        launch_testing(model, param)

    del train_dataloader, valid_dataloader, model, optimizer, loss
    if cuda_is_available():
        torch.cuda.empty_cache()


def load_experiment_assets(param):
    CometLogger.print("~~ Loading dataset's dataloaders ~~")
    train_dataloader, valid_dataloader = load_dataset_dataloaders(param)
    CometLogger.print("~~ Loading the model ~~")
    model = load_model(param)
    CometLogger.print("~~ Loading the optimizer ~~")
    optimizer = load_optimizer(param, model)
    CometLogger.print("~~ Loading the loss ~~")
    loss = load_loss(param)
    return loss, model, optimizer, train_dataloader, valid_dataloader


def setup_comet_experiment(experiment, param, repo):
    updated_params = dict()
    # Replace parameters using the experiment params, used in hyperparameter search
    if len(experiment.params) > 0:
        for k, v in experiment.params.items():
            if k in param.get_params_as_dict():
                param.get_params_as_dict()[k] = v
                updated_params[k] = v

    model_code = open("./Models/{}.py".format(param.model), "r").read()
    loss_code = open("./Models/Losses.py", "r").read()
    code = model_code + "\n###############\n" + loss_code
    experiment.set_code(code, True)

    experiment.log_parameters(param.get_params_as_dict())

    subject = repo.git.log("-1", "--pretty=%B").replace("\n", " - ")
    experiment_name = subject
    if len(updated_params) > 0:
        experiment_name = experiment_name + " - " + ", ".join(
            ["{}: {}".format(k, v) for k, v in updated_params.items()])
    experiment.set_name(experiment_name)


def launch_parallel_experiment(gpu_rank, api_key, experiment_keys, experiment_params, repo_path):
    torch.cuda.set_device(gpu_rank)
    param = Parameters()
    param.segment_dataset = False
    param.model_backup_destination = param.model_backup_destination + "/process_{}".format(gpu_rank)
    experiment = ExistingExperiment(api_key=api_key,
                                    previous_experiment=experiment_keys[gpu_rank],
                                    log_env_details=True,
                                    log_env_gpu=True,
                                    log_env_cpu=True)
    experiment.params = experiment_params[gpu_rank]
    repo = Repo(repo_path)

    with CometLogger(experiment, gpu_id=gpu_rank, print_to_comet_only=True):
        setup_comet_experiment(experiment, param, repo)
        CometLogger.print("-> loading experiments assets:")
        loss, model, optimizer, train_dataloader, valid_dataloader = load_experiment_assets(param)

        if param.train:
            CometLogger.print("~~ Launching the training ~~")
            CometLogger.print("Sleeping {} secs to reduce chances of deadlock.".format(gpu_rank))
            sleep(gpu_rank)

            launch_training(model, train_dataloader, valid_dataloader, optimizer, loss, param)
        if param.test:
            CometLogger.print("~~ Testing the model ~~")
            launch_testing(model, param)

    del train_dataloader, valid_dataloader, model, optimizer, loss
    torch.cuda.empty_cache()


def launch_experiment(experiments: list, repo: Repo):
    param = Parameters()

    # The datasets needs to be segmented before any experiments is launched to prevent process conflicts
    if param.segment_dataset:
        Loaders.segment_datasets(param)

    world_size = torch.cuda.device_count()

    if cuda_is_available() and world_size > 1 and 1 < len(experiments) <= world_size:
        print("-> Launching {} parallel experiments...".format(torch.cuda.device_count()))
        experiment_keys = [experiment.get_key() for experiment in experiments]
        print("-> experiment keys: {}".format(experiment_keys))
        experiment_params = [experiment.params for experiment in experiments]
        api_key = experiments[0].api_key

        print("-> spawning the experiments' processes")
        multiprocessing.spawn(launch_parallel_experiment, nprocs=len(experiments), args=(api_key,
                                                                                   experiment_keys,
                                                                                   experiment_params,
                                                                                   repo.git_dir))
    elif len(experiments) == 1:
        with CometLogger(experiments[0]):
            print("-> launching single experiment")
            launch_single_GPU_experiment(experiments[0], repo, param)
    else:
        raise NotImplementedError()


if __name__ == '__main__':

    repo = Repo("./")
    assert not repo.bare

    project_name = "candidate-tests"

    if len(sys.argv) > 1:
        # get the config file from the arguments
        print("-> Loading the optimizer...")
        opt = CometOptimizer(config=sys.argv[1])

        active_parallel_experiements = []
        for experiment in opt.get_experiments(project_name=project_name,
                                              workspace="olibd"):
            print("-> Registering experiment {} with Comet...".format(experiment.get_key()))
            active_parallel_experiements.append(experiment)

            if len(active_parallel_experiements) == torch.cuda.device_count() or (not cuda_is_available() and len(active_parallel_experiements) == 1):
                launch_experiment(active_parallel_experiements, repo)
                active_parallel_experiements = []

        # If the last batch of experiments had a lower experiment count
        # than the number of GPUs, then it hasn't run yet. So we need to
        # run them now.
        if len(active_parallel_experiements) > 0:
            launch_experiment(active_parallel_experiements, repo)

    else:
        experiment = Experiment(project_name=project_name,
                                workspace="olibd")
        launch_experiment([experiment], repo)
