from typing import Any, Dict, Tuple

import torch
from torch.nn import functional as F
from functools import partial
from lightning import LightningModule
import itertools
from PIL import Image
from torchvision import transforms

from src.models.components.gan import Generator, Discriminator


class GANLitModule(LightningModule):

    def __init__(
        self,
        max_epoch=100,
        sample_image_a="data/summer2winter_yosemite/testA/2010-09-07 12:23:20.jpg",
        sample_image_b="data/summer2winter_yosemite/testB/2006-04-11 11:21:20.jpg",
        img_size=256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.sample_image_a = sample_image_a
        self.sample_image_b = sample_image_b

        self.transformer = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.max_epoch = max_epoch

        self.generator_a_2_b = Generator()
        self.generator_b_2_a = Generator()

        self.discriminator_a_2_b = Discriminator()
        self.discriminator_b_2_a = Discriminator()

        self.automatic_optimization = False

    def process_sample_img(self, img_path, device=None):
        img = Image.open(img_path)
        img: torch.Tensor = self.transformer(img)
        img = img.unsqueeze(0)
        if device is not None:
            img = img.to(device)
        return img

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return x

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:

        image_a = batch["image_a"]
        image_b = batch["image_b"]
        batch_size = image_a.shape[0]

        optimizer_g_en, optimizer_d_a_2_b, optimizer_d_b_2_a = self.optimizers()

        #### 训练生成器
        self.toggle_optimizer(optimizer_g_en)

        # ID 一致性损失
        output_a_2_a = self.generator_b_2_a(image_a)

        loss_a_2_a = F.l1_loss(output_a_2_a, image_a) * 5

        output_b_2_b = self.generator_a_2_b(image_b)

        loss_b_2_b = F.l1_loss(output_b_2_b, image_b) * 5

        # 对抗 损失

        output_a_2_b = self.generator_a_2_b(image_a)

        labels_real = (
            torch.empty(size=(batch_size, 1)).uniform_(0.9, 1.1).type_as(image_a)
        )

        loss_a_2_b = F.mse_loss(self.discriminator_a_2_b(output_a_2_b), labels_real)

        output_b_2_a = self.generator_b_2_a(image_b)
        loss_b_2_a = F.mse_loss(self.discriminator_b_2_a(output_b_2_a), labels_real)

        # 循环一致性损失

        output_a_2_b_2_a = self.generator_b_2_a(output_a_2_b)

        loss_cycle_a_b_a = F.l1_loss(output_a_2_b_2_a, image_a) * 10

        output_b_a_b = self.generator_a_2_b(output_b_2_a)

        loss_cycle_b_a_b = F.l1_loss(output_b_a_b, image_b) * 10

        loss_total_generator = (
            loss_a_2_a
            + loss_b_2_b
            + loss_a_2_b
            + loss_b_2_a
            + loss_cycle_a_b_a
            + loss_cycle_b_a_b
        )
        self.manual_backward(loss_total_generator)
        optimizer_g_en.step()
        optimizer_g_en.zero_grad()
        self.untoggle_optimizer(optimizer_g_en)

        #### 训练判别器 a-> b

        self.toggle_optimizer(optimizer_d_a_2_b)

        discriminator__a_b_output = self.discriminator_a_2_b(output_a_2_b.detach())
        label_fake = (
            torch.empty(size=(batch_size, 1)).type_as(image_a).uniform_(-0.1, 0.1)
        )
        loss_d_a_b = F.mse_loss(discriminator__a_b_output, label_fake)

        self.manual_backward(loss_d_a_b)
        optimizer_d_a_2_b.step()
        optimizer_d_a_2_b.zero_grad()
        self.untoggle_optimizer(optimizer_d_a_2_b)

        #### 训练判别器b -> a

        self.toggle_optimizer(optimizer_d_b_2_a)

        discriminator_b_a_output = self.discriminator_b_2_a(output_b_2_a.detach())

        loss_d_b_a = F.mse_loss(discriminator_b_a_output, label_fake)

        self.manual_backward(loss_d_b_a)
        optimizer_d_b_2_a.step()
        optimizer_d_b_2_a.zero_grad()
        self.untoggle_optimizer(optimizer_d_b_2_a)

    
        current_lr = optimizer_g_en.param_groups[0]['lr']
        loss_dict = dict(
            loss_a_2_a=loss_a_2_a,
            loss_b_2_b=loss_b_2_b,
            loss_a_2_b=loss_a_2_b,
            loss_b_2_a=loss_b_2_a,
            loss_cycle_a_b_a=loss_cycle_a_b_a,
            loss_cycle_b_a_b=loss_cycle_b_a_b,
            loss_total_generator=loss_total_generator,
            loss_d_a_b=loss_d_a_b,
            loss_d_b_a=loss_d_b_a,
            lr=current_lr
        )
        
        self.log_dict(loss_dict, prog_bar=True, logger=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        schedules = self.lr_schedulers()
        for schedule in schedules:
            schedule.step()

    def norm_image(self, x):
        x = x * 0.5 + 0.5
        return x

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        device = next(iter(self.generator_a_2_b.parameters())).device

        sample_a = self.process_sample_img(self.sample_image_a, device=device)
        sample_a_2_b = self.generator_a_2_b(sample_a)
        sample_a_2_b = self.norm_image(sample_a_2_b)

        sample_b = self.process_sample_img(self.sample_image_b, device=device)
        sample_b_2_a = self.generator_b_2_a(sample_b)
        sample_b_2_a = self.norm_image(sample_b_2_a)
        
        
        
        self.logger.experiment.add_image(
            "train/sample_a_2_b", sample_a_2_b[0], self.current_epoch
        )
        self.logger.experiment.add_image(
            "train/sample_b_2_a", sample_b_2_a[0], self.current_epoch
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_g_en = torch.optim.AdamW(
            itertools.chain(
                self.generator_a_2_b.parameters(), self.generator_b_2_a.parameters()
            ),
            lr=1e-3,
        )

        optimizer_d_a_2_b = torch.optim.AdamW(
            self.discriminator_a_2_b.parameters(), lr=1e-3
        )
        optimizer_d_b_2_a = torch.optim.AdamW(
            self.discriminator_b_2_a.parameters(), lr=1e-3
        )

        schedule_fn = partial(
            torch.optim.lr_scheduler.CosineAnnealingLR, T_max=self.max_epoch, eta_min=0
        )
        
        # 将调度器包装在字典中，并指定 interval='epoch'
        scheduler_g_en = {
            'scheduler': schedule_fn(optimizer=optimizer_g_en),
            'interval': 'epoch',  # <--- 关键改动
            'frequency': 1
        }
        scheduler_d_a_2_b = {
            'scheduler': schedule_fn(optimizer=optimizer_d_a_2_b),
            'interval': 'epoch',  # <--- 关键改动
            'frequency': 1
        }
        scheduler_d_b_2_a = {
            'scheduler': schedule_fn(optimizer=optimizer_d_b_2_a),
            'interval': 'epoch',  # <--- 关键改动
            'frequency': 1
        }

        return [
            optimizer_g_en,
            optimizer_d_a_2_b,
            optimizer_d_b_2_a,
        ], [
            scheduler_g_en,
            scheduler_d_a_2_b,
            scheduler_d_b_2_a,
        ]


if __name__ == "__main__":
    pass
