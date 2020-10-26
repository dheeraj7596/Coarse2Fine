import torch
import scipy.special
import numpy as np
from torch.autograd import Variable


class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """

    @staticmethod
    def forward(ctx, k, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        m = 100
        ctx.save_for_backward(k)
        k = k.double()
        answer = (m / 2 - 1) * torch.log(k) - torch.log(scipy.special.ive(m / 2 - 1, k)).to(device) - k - (
                m / 2) * np.log(2 * np.pi)
        # answer = (m / 2 - 1) * torch.log(k) - torch.log(scipy.special.ive(m / 2 - 1, k)).to(device) - (m / 2) * np.log(2 * np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output, device):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        m = 100
        # x = -ratio(m/2, k)
        k = k.double()
        x = -((scipy.special.ive(m / 2, k)) / (scipy.special.ive(m / 2 - 1, k))).to(device)
        x = x.float()

        return grad_output * Variable(x)


def contrastiveNLLvMF(outputs, targets, label_embeddings, device):
    """
    :param outputs: BERT 100 dim vectors
    :param targets: label indices
    :param label_embeddings: tensor with label embeddings at their corresponding label indices
    :param device: device
    :return: loss, logits
    """
    loss = 0
    logits = []
    batch_size = outputs.size(0)
    logcmk = Logcmk.apply

    if targets is not None:
        for i, (out_t, targ_t) in enumerate(zip(outputs, targets)):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            num = 0
            den = 0
            logits_temp = []
            for l in range(label_embeddings.shape[0]):
                tar_vec_t = label_embeddings[l]
                tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
                n_log_vmf = - logcmk(kappa, device) + torch.log(1 + kappa) * (
                        0.2 - (out_vec_norm_t * tar_vec_norm_t).sum(dim=-1))
                # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
                vmf = torch.exp(-n_log_vmf)
                if l == targ_t:
                    num = vmf
                else:
                    den += vmf
                logits_temp.append(vmf)
            loss += -torch.log(num / den)
            logits.append(logits_temp)

        loss = loss.div(batch_size)
        logits = torch.tensor(logits).to(device)
    else:
        for i, out_t in enumerate(outputs):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            logits_temp = []
            for l in range(label_embeddings.shape[0]):
                tar_vec_t = label_embeddings[l]
                tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
                n_log_vmf = - logcmk(kappa, device) + torch.log(1 + kappa) * (
                        0.2 - (out_vec_norm_t * tar_vec_norm_t).sum(dim=-1))
                # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
                vmf = torch.exp(-n_log_vmf)
                logits_temp.append(vmf)
            logits.append(logits_temp)

        logits = torch.tensor(logits).to(device)
    return loss, logits


if __name__ == "__main__":
    outputs = torch.randn(5, 32)
    label_embeddings = [np.random.randn(32), np.random.randn(32), np.random.randn(32)]
    label_embeddings = torch.tensor(np.array(label_embeddings))
    targets = [0, 2, 1, 1, 0]
    t = contrastiveNLLvMF(outputs, None, label_embeddings, device=torch.device("cpu"))
