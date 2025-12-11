from typing import Dict
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


class Trainer:
    def __init__(
        self,
        epoch_num,
        train_batch_size,
        test_batch_size,
        lr,
        weight_decay=1e-5,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.epoch_num = epoch_num
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.weight_decay = weight_decay

    def setup(
        self,
        model,
        data_train,
        data_tests,
        model_configs: Dict = {},
        log_dir: str = "./exp/logs",
    ):
        torch.manual_seed(1)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.device = self.device

        # setup model_configs
        for att in model_configs.keys():
            if hasattr(self.model, att):
                setattr(self.model, att, model_configs[att])

        # setup train object
        self.optimizer = AdamW(
            params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.7)

        # setup tensorboard
        self.writer = SummaryWriter(log_dir)

        # set data info
        self.data_train = data_train
        self.data_tests = data_tests

        # setup dataloader
        if data_train:
            self.dataloader_train = DataLoader(
                data_train,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=10,
                collate_fn=data_train.collate_fn,
            )
        else:
            self.dataloader_train = None
        self.dataloader_tests = {
            data_test_name: DataLoader(
                data_test,
                batch_size=self.test_batch_size,
                shuffle=False,
                num_workers=10,
                collate_fn=data_test.collate_fn,
            )
            for data_test_name, data_test in data_tests.items()
        }

    def train(self, early_stopping_params=None, checkpoint_dir="./exp/models"):
        os.makedirs(checkpoint_dir, exist_ok=True)

        # init
        self.step = 0

        # early stopping params
        if early_stopping_params is not None:
            last_loss = 999999
            trigger_times = 0

        for epoch in range(1, self.epoch_num + 1):
            # pre epoch
            self.model.pre_epoch(self, epoch)

            # train epoch
            log_info = self.model.train_epoch(self, epoch)

            # log info
            for step_loss in log_info["step_losses"]:
                self.step += 1
                self.writer.add_scalar("TrainInfo/step_loss", step_loss, self.step)
            self.writer.add_scalar(
                "TrainInfo/epoch_loss", log_info["epoch_loss"], epoch
            )
            self.writer.add_scalar(
                "TrainInfo/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            for other_info_key in log_info["others"].keys():
                self.writer.add_scalar(
                    f"TrainInfo/{other_info_key}",
                    log_info["others"][other_info_key],
                    epoch,
                )

            # save model
            self.model.save(os.path.join(checkpoint_dir, f"epoch_{epoch:03d}"))

            # update lr
            self.scheduler.step()

            # test
            if epoch % 5 == 0:
                test_log_info = {}
                for data_type in self.dataloader_tests.keys():
                    test_log_info[data_type] = self.test(
                        self.dataloader_tests[data_type],
                        epoch,
                        data_name=f"{data_type}",
                    )

            # post epoch
            self.model.post_epoch(self, epoch)

            # early stopping
            if early_stopping_params is not None:
                current_loss = test_log_info[early_stopping_params["data_name"]][
                    early_stopping_params["loss_type"]
                ]

                if current_loss > last_loss:
                    trigger_times += 1

                last_loss = current_loss

                if trigger_times >= early_stopping_params["patience"]:
                    print("Early stopping!")
                    break

        # make sure that all pending events have been written to disk
        self.writer.flush()

    def test(self, dataloader, epoch, data_name):
        log_info = self.model.test_epoch(self, dataloader, epoch, data_name)
        for log_key in log_info.keys():
            self.writer.add_scalars(
                f"TestInfo/{log_key}", {data_name: log_info[log_key]}, epoch
            )
        return log_info
