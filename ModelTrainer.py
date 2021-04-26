import time

import torch
from stopit import ThreadingTimeout
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Common.Helpers import cuda_is_available
from Logging import CometLogger
from Models.Losses import AbstractLoss, BatchSegmentMSELoss
from pytorchtools import EarlyStopping


class ModelTrainer:
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, valid_dataloader, optimizer: Optimizer,
                 loss: AbstractLoss, early_stopping_patience=7, model_backup_destination="./", resume=False,
                 gradient_clipping_value=None):
        self.model: nn.Module = model
        self.train_dataloader: DataLoader = train_dataloader
        self.valid_dataloader: DataLoader = valid_dataloader
        self.optimizer: Optimizer = optimizer
        # Loss used for benchmarking agaisnt other runs only in case the loss function from which backprop is computed changes
        self.benchmark_MSE_loss: AbstractLoss = BatchSegmentMSELoss()
        # Custom loss is used for backpropagating
        self.custom_loss: AbstractLoss = loss
        self.gradient_clipping_value = gradient_clipping_value
        self.model_backup_destination = self._get_backup_destination(model_backup_destination,
                                                                     model,
                                                                     train_dataloader,
                                                                     optimizer,
                                                                     loss)
        self.early_stopper = EarlyStopping(patience=early_stopping_patience,
                                           verbose=True,
                                           destination_path=self.model_backup_destination)

        if resume:
            CometLogger.print("Resuming the training of {}".format(self.model_backup_destination))
            CometLogger.print("Overriding the Model and Optimizer's state dictionaries with the checkpoint's dicts")
            self.model.load_state_dict(self.early_stopper.load_model_checkpoint())
            self.optimizer.load_state_dict(self.early_stopper.load_optimizer_checkpoint())

    def run(self, epochs_number: int) -> nn.Module:
        for epoch in self._epochs(epochs_number):
            CometLogger.print("=========== Epoch {} ===========".format(epoch))
            t0 = time.time()
            custom_train_loss, train_benchmark_loss = self._train()
            custom_valid_loss, valid_benchmark_loss = self._validate()
            t1 = time.time()
            epoch_run_time = t1-t0
            self._log_epoch(custom_train_loss, custom_valid_loss, epoch, epoch_run_time, train_benchmark_loss,
                           valid_benchmark_loss)
            self.early_stopper(custom_valid_loss, self.model, self.optimizer)

            if self.early_stopper.early_stop:
                CometLogger.get_experiment().log_metric("Early stop epoch", epoch + 1)
                CometLogger.print("Early stopping")
                break

        CometLogger.print("Training complete, loading the last early stopping checkpoint to memory...")
        self.model.load_state_dict(self.early_stopper.load_model_checkpoint())
        return self.model

    def _log_epoch(self, custom_train_loss, custom_valid_loss, epoch, epoch_run_time, train_benchmark_loss,
                  valid_benchmark_loss):
        CometLogger.print("Epoch run time: {}".format(epoch_run_time))
        CometLogger.get_experiment().log_metric("epoch run time", epoch_run_time, epoch=epoch)
        CometLogger.get_experiment().log_metric("mean training loss", train_benchmark_loss, epoch=epoch)
        CometLogger.get_experiment().log_metric("mean validation loss", valid_benchmark_loss, epoch=epoch)
        CometLogger.get_experiment().log_metric("custom mean training loss", custom_train_loss, epoch=epoch)
        CometLogger.get_experiment().log_metric("custom mean validation loss", custom_valid_loss, epoch=epoch)
        CometLogger.print("Mean train loss: {}, Valid train loss: {}".format(custom_train_loss, custom_valid_loss))
        CometLogger.get_experiment().log_metric("epoch", epoch)
        CometLogger.get_experiment().log_epoch_end(epoch_cnt=epoch)

    def _train(self) -> tuple:
        timer_start_time = time.time()

        self.model.train()

        losses_sum = 0
        benchmark_losses_sum = 0

        for i, (input, target) in enumerate(self.train_dataloader):
            CometLogger.get_experiment().log_metric("Current batch", i + 1)
            CometLogger.get_experiment().log_metric("Total nbr of batches", len(self.train_dataloader))

            # Only log this if we are NOT in a multiprocessing session
            if CometLogger.gpu_id is None:
                print("--> processing batch {}/{} of size {}".format(i+1, len(self.train_dataloader), len(input)))
            if cuda_is_available():
                with ThreadingTimeout(14400.0) as timeout_ctx1:
                    input = input.cuda(non_blocking=self.train_dataloader.pin_memory)
                    target = target.cuda(non_blocking=self.train_dataloader.pin_memory)
                if not bool(timeout_ctx1):
                    CometLogger.fatalprint('Encountered fatally long delay when moving tensors to GPUs')


            prediction = self.model.forward(input)


            with ThreadingTimeout(14400.0) as timeout_ctx3:
                if type(prediction) is tuple:
                    benchmark_loss = self.benchmark_MSE_loss.compute(prediction[0], target)
                else:
                    benchmark_loss = self.benchmark_MSE_loss.compute(prediction, target)
            if not bool(timeout_ctx3):
                CometLogger.fatalprint('Encountered fatally long delay during computation of benchmark loss')

            with ThreadingTimeout(14400.0) as timeout_ctx4:
                benchmark_losses_sum += float(benchmark_loss.data.cpu().numpy())
            if not bool(timeout_ctx4):
                CometLogger.fatalprint('Encountered fatally long delay during summation of benchmark losses')

            with ThreadingTimeout(14400.0) as timeout_ctx4:
                loss = self.custom_loss.compute(prediction, target)
            if not bool(timeout_ctx4):
                CometLogger.fatalprint('Encountered fatally long delay during computation of the custom loss')

            self._backpropagate(loss)

            with ThreadingTimeout(14400.0) as timeout_ctx6:
                losses_sum += float(loss.data.cpu().numpy())
            if not bool(timeout_ctx6):
                CometLogger.fatalprint('Encountered fatally long delay during loss addition')

        timer_end_time = time.time()

        CometLogger.get_experiment().log_metric("Epoch training time", timer_end_time - timer_start_time)

        return losses_sum/len(self.train_dataloader), benchmark_losses_sum/len(self.train_dataloader)

    def _validate(self) -> tuple:
        self.model.eval()

        losses_sum = 0
        benchmark_losses_sum = 0
        for input, target in self.valid_dataloader:
            if cuda_is_available():
                input = input.cuda(non_blocking=self.valid_dataloader.pin_memory)
                target = target.cuda(non_blocking=self.valid_dataloader.pin_memory)

            prediction = self.model.forward(input)
            if type(prediction) is tuple:
                benchmark_loss = self.benchmark_MSE_loss.compute(prediction[0], target)
            else:
                benchmark_loss = self.benchmark_MSE_loss.compute(prediction, target)

            benchmark_losses_sum += float(benchmark_loss.data.cpu().numpy())
            loss = self.custom_loss.compute(prediction, target)
            losses_sum += float(loss.data.cpu().numpy())

        return losses_sum / len(self.valid_dataloader), benchmark_losses_sum/len(self.train_dataloader)

    def _backpropagate(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping_value is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_value)
        self.optimizer.step()

    def _epochs(self, epochs_numbers):
        return range(0, epochs_numbers)

    def _get_backup_destination(self, base_path, model: nn.Module, dataloader: DataLoader, optimizer: Optimizer,
                                loss: AbstractLoss):

        return "{}/{}_{}_{}_{}".format(base_path,
                                    type(model).__name__,
                                    type(dataloader.dataset).__name__,
                                    type(optimizer).__name__,
                                    type(loss).__name__)
