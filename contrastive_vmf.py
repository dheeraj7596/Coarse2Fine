import torch
import scipy.special
import numpy as np
from torch.autograd import Variable


class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """

    @staticmethod
    def forward(ctx, k):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        m = 100
        ctx.save_for_backward(k)
        k = k.double()
        answer = (m / 2 - 1) * torch.log(k) - torch.log(scipy.special.ive(m / 2 - 1, k.cpu())) - k - (m / 2) * np.log(
            2 * np.pi)
        # answer = (m / 2 - 1) * torch.log(k) - torch.log(scipy.special.ive(m / 2 - 1, k)).to(device) - k - (m / 2) * torch.tensor(np.log(2 * np.pi))
        # answer = (m / 2 - 1) * torch.log(k)
        # answer = answer - torch.log(torch.tensor(scipy.special.ive(m / 2 - 1, k)))
        # answer = answer - k - (m / 2) * torch.tensor(np.log(2 * np.pi))
        # answer = (m / 2 - 1) * torch.log(k) - torch.log(scipy.special.ive(m / 2 - 1, k)).to(device) - (m / 2) * np.log(2 * np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        m = 100
        # x = -ratio(m/2, k)
        k = k.double()
        x = -((scipy.special.ive(m / 2, k.cpu())) / (scipy.special.ive(m / 2 - 1, k.cpu())))
        x = x.float()

        return grad_output * Variable(x)


# def contrastiveNLLvMF(outputs, targets, label_embeddings, device):
#     """
#         Training: Flat training
#         Prediction: Flat prediction
#     :param outputs: BERT 100 dim vectors
#     :param targets: label indices
#     :param label_embeddings: tensor with label embeddings at their corresponding label indices
#     :param device: device
#     :return: loss, logits
#     """
#     loss = 0
#     logits = []
#     batch_size = outputs.size(0)
#     logcmk = Logcmk.apply
#
#     if targets is not None:
#         for i, (out_t, targ_t) in enumerate(zip(outputs, targets)):
#             # out_t -> dim
#             # targ_t -> label_index
#
#             out_vec_t = out_t
#             kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
#             out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)
#
#             temp = []
#             left = 0
#             logits_temp = []
#             for l in range(label_embeddings.shape[0]):
#                 tar_vec_t = label_embeddings[l]
#                 tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
#                 n_log_vmf = - logcmk(kappa) + torch.log(1 + kappa) * (
#                         0.2 - (out_vec_norm_t * tar_vec_norm_t).sum(dim=-1))
#                 # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
#                 if l == targ_t:
#                     left = n_log_vmf
#                 else:
#                     temp.append(-n_log_vmf)
#                 logits_temp.append(-n_log_vmf)
#             right = torch.logsumexp(torch.tensor(temp).to(device).view(1, -1), dim=1).to(device)
#             loss += (left + right)
#             logits.append(logits_temp)
#
#         loss = loss.div(batch_size).to(device)
#         print("Loss:", loss, flush=True)
#         logits = torch.tensor(logits).to(device)
#     else:
#         for i, out_t in enumerate(outputs):
#             # out_t -> dim
#             # targ_t -> label_index
#
#             out_vec_t = out_t
#             kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
#             out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)
#
#             logits_temp = []
#             for l in range(label_embeddings.shape[0]):
#                 tar_vec_t = label_embeddings[l]
#                 tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
#                 n_log_vmf = - logcmk(kappa) + torch.log(1 + kappa) * (
#                         0.2 - (out_vec_norm_t * tar_vec_norm_t).sum(dim=-1))
#                 # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
#                 logits_temp.append(-n_log_vmf)
#             logits.append(logits_temp)
#
#         logits = torch.tensor(logits).to(device)
#     return loss, logits


def contrastiveNLLvMF(outputs, targets, label_embeddings, device, additional_args):
    """
        Training: Trains over coarse-grained labels contrasted over other coarse-grained labels and their fine-grained labels.
        prediction: Predicts over all child labels.
    :param outputs: BERT 100 dim vectors
    :param targets: label indices
    :param label_embeddings: tensor with label embeddings at their corresponding label indices
    :param device: device
    :param additional_args: contain possible_labels and contrastive map.
                possible_labels: possible labels indicated with their label indices
                contrastive_map: contrastive map from one label index to other label indices with which it has to be contrasted.
    :return: loss, logits
    """
    loss = 0
    logits = []
    batch_size = outputs.size(0)
    logcmk = Logcmk.apply
    intra_label_index = {}
    possible_labels = additional_args["possible_labels"]
    contrastive_map = additional_args["contrastive_map"]

    for i, pos_lbl in enumerate(possible_labels):
        intra_label_index[pos_lbl] = i

    if targets is not None:
        for i, (out_t, targ_t) in enumerate(zip(outputs, targets)):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            temp = []
            logits_temp = [0] * len(possible_labels)
            left = compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, targ_t.item())
            logits_temp[intra_label_index[targ_t.item()]] = -left

            for con_label in contrastive_map[targ_t.item()]:
                n_log_vmf = compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, con_label)
                # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
                temp.append(-n_log_vmf)
                try:
                    logits_temp[intra_label_index[con_label]] = -n_log_vmf
                except:
                    continue

            right = torch.logsumexp(torch.tensor(temp).to(device).view(1, -1), dim=1).to(device)
            loss += (left + right)
            logits.append(logits_temp)

        loss = loss.div(batch_size).to(device)
        print("Loss:", loss, flush=True)
        logits = torch.tensor(logits).to(device)
    else:
        for i, out_t in enumerate(outputs):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            logits_temp = []

            for l in possible_labels:
                n_log_vmf = compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, l)
                # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
                logits_temp.append(-n_log_vmf)
            logits.append(logits_temp)

        logits = torch.tensor(logits).to(device)
    return loss, logits


def contrastiveNLLvMF_self(outputs, targets, label_embeddings, device, additional_args):
    """
        Training: Trains over coarse-grained labels contrasted over other coarse-grained labels and their fine-grained labels.
        prediction: Predicts over all child labels.
    :param outputs: BERT 100 dim vectors
    :param targets: label indices
    :param label_embeddings: tensor with label embeddings at their corresponding label indices
    :param device: device
    :param additional_args: contain possible_labels and contrastive map.
                possible_labels: possible labels indicated with their label indices
                contrastive_map: contrastive map from one label index to other label indices with which it has to be contrasted.
    :return: loss, logits
    """
    loss = 0
    logits = []
    batch_size = outputs.size(0)
    logcmk = Logcmk.apply
    possible_labels = additional_args["possible_labels"]

    if targets is not None:
        for i, (out_t, targ_t) in enumerate(zip(outputs, targets)):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            temp = []
            logits_temp = []
            for con_label in possible_labels:
                n_log_vmf = compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, con_label)
                temp.append(-n_log_vmf)
                logits_temp.append(-n_log_vmf)

            left = torch.max(torch.tensor(temp))
            ind = temp.index(left)
            del temp[ind]
            right = torch.logsumexp(torch.tensor(temp).to(device).view(1, -1), dim=1).to(device)
            loss += (left + right)
            logits.append(logits_temp)

        loss = loss.div(batch_size).to(device)
        print("Loss:", loss, flush=True)
        logits = torch.tensor(logits).to(device)
    else:
        for i, out_t in enumerate(outputs):
            # out_t -> dim
            # targ_t -> label_index

            out_vec_t = out_t
            kappa = out_vec_t.norm(p=2, dim=-1)  # *tar_vec_t.norm(p=2,dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_vec_t, p=2, dim=-1)

            logits_temp = []

            for l in possible_labels:
                n_log_vmf = compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, l)
                # n_log_vmf = - logcmk(kappa, device) - (out_vec_t * tar_vec_norm_t).sum(dim=-1)
                logits_temp.append(-n_log_vmf)
            logits.append(logits_temp)

        logits = torch.tensor(logits).to(device)
    return loss, logits


def compute_n_log_vmf(kappa, label_embeddings, logcmk, out_vec_norm_t, targ_t_index):
    tar_vec_t = label_embeddings[targ_t_index]
    tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
    n_log_vmf = - logcmk(kappa) + torch.log(1 + kappa) * (0.2 - (out_vec_norm_t * tar_vec_norm_t).sum(dim=-1))
    return n_log_vmf


if __name__ == "__main__":
    outputs = torch.randn(5, 32)
    label_embeddings = [np.random.randn(32), np.random.randn(32), np.random.randn(32)]
    label_embeddings = torch.tensor(np.array(label_embeddings))
    targets = [0, 2, 1, 1, 0]
    t = contrastiveNLLvMF(outputs, targets, label_embeddings, device=torch.device("cpu"))
