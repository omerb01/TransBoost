import torch


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return idx, image, label


class TransductiveDatasetTest(torch.utils.data.Dataset):
    def __init__(self, dataset_test, pseudo_labels, confidence):
        self.dataset_test = dataset_test
        self.pseudo_labels = pseudo_labels
        self.confidence = confidence

    def __len__(self):
        return len(self.dataset_test)

    def __getitem__(self, idx):
        image, _ = self.dataset_test[idx]
        return image, self.pseudo_labels[idx].item(), self.confidence[idx].item()
