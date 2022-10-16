import torch

from pl_bolts.datamodules.imagenet_datamodule import ImagenetDataModule as ImagenetBoltsDataModule
from pl_bolts.datasets.imagenet_dataset import UnlabeledImagenet


class ImagenetDataModule(ImagenetBoltsDataModule):
    def __init__(
            self,
            data_dir,
            meta_dir=None,
            val_split=0,
            image_size=224,
            num_workers=0,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            seed=42,
            *args,
            **kwargs,
    ) -> None:
        super(ImagenetDataModule, self).__init__(data_dir=data_dir,
                                                 meta_dir=meta_dir,
                                                 num_imgs_per_val_class=val_split,
                                                 image_size=image_size,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 pin_memory=pin_memory,
                                                 drop_last=drop_last,
                                                 *args,
                                                 **kwargs)
        self.seed = seed
        self.train_transforms = self.train_transform()
        self.val_transforms = self.val_transform()
        self.test_transforms = self.val_transform()

    def train_dataset(self, transforms):
        return UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=-1,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="train",
            transform=transforms,
        )

    def val_dataset(self, transforms):
        return UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class_val_split=self.num_imgs_per_val_class,
            meta_dir=self.meta_dir,
            split="val",
            transform=transforms,
        )

    def test_dataset(self, transforms):
        return UnlabeledImagenet(
            self.data_dir, num_imgs_per_class=-1, meta_dir=self.meta_dir, split="test", transform=transforms
        )

    def train_dataloader(self):
        train_transforms = self.train_transform() if self.train_transforms is None else self.train_transforms
        return torch.utils.data.DataLoader(
            self.train_dataset(train_transforms),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        val_transforms = self.val_transform() if self.val_transforms is None else self.val_transforms
        return torch.utils.data.DataLoader(
            self.val_dataset(val_transforms),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        test_transforms = self.val_transform() if self.test_transforms is None else self.test_transforms
        return torch.utils.data.DataLoader(
            self.test_dataset(test_transforms),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
