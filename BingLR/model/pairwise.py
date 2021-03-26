import torch
from torch import nn

def pairwise_tensor(x, list_length):
    assert len(x.shape) == 2
    list_length = x.shape[1]
    y = torch.unsqueeze(x, -1)  # (BS/ListLength,ListLength,1)
    x = torch.unsqueeze(x, 1)  # (BS/ListLength,1,ListLength)
    x = x.repeat(1, list_length, 1)  # (BS/ListLength,ListLength,ListLength)
    y = y.repeat(1, 1, list_length)  # (BS/ListLength,ListLength,ListLength)
    y = torch.unsqueeze(y, -1)  # (BS/ListLength,ListLength,ListLength,1)
    x = torch.unsqueeze(x, -1)  # (BS/ListLength,ListLength,ListLength,1)
    z = torch.cat((y, x), -1)  # (BS/ListLength,ListLength,ListLength,2)
    return z

def _safe_mean(losses, num_present):
    """Computes a safe mean of the losses.
    Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.
    Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
    """
    total_loss = torch.sum(losses)
    if num_present == 0:
        return 0 * total_loss
    else:
        return torch.div(total_loss, num_present)

def _num_elements(losses):
    """Computes the number of elements in `losses` tensor."""
    return torch.tensor(losses.size()[0]).type(losses.dtype)

def _num_present(losses, weights, per_batch=False):
    if ((isinstance(weights, float) and weights != 0.0) or
            (weights.dim() == 0 and not torch.equal(weights, torch.tensor(0.0)))):
        return _num_elements(losses)
    else:
        return torch.sum(weights)

def compute_weighted_loss(losses, weights, reduction='sum_by_nonzero_weight'):
    input_dtype = losses.dtype
    losses = losses.float()
    weights = weights.float()
    weighted_loss = torch.mul(losses, weights)
    if reduction == None:
        loss = weighted_loss
    else:
        loss = torch.sum(weighted_loss)
        if reduction == 'mean':
            loss = _safe_mean(loss, torch.sum(torch.ones_like(losses) * weights))
        elif reduction == 'sum_by_nonzero_weight':
            loss = _safe_mean(loss, _num_present(losses, weights))
        else:
            loss = _safe_mean(loss, _num_elements(losses))
    loss = loss.type(input_dtype)
    return loss

def PairwiseHRSLoss(x, label, list_length):
    pws_model_logits = pairwise_tensor(x.view(-1, list_length), list_length)
    pws_label = pairwise_tensor(label.view(-1, list_length), list_length)
    pws_label = torch.cat((torch.unsqueeze(pws_label[:, :, :, 0] > pws_label[:, :, :, 1], -1).float(),
                               torch.unsqueeze(pws_label[:, :, :, 0] < pws_label[:, :, :, 1], -1).float()), -1)
    pws_model_scores = torch.nn.functional.log_softmax(pws_model_logits,
                                                           -1)  # (BS/ListLength,ListLength,ListLength,2)

    loss = -(pws_model_scores * pws_label).sum(-1)  # (BS/ListLength,ListLength,ListLength)
    pws_hrs_loss = loss.sum(-1).view(-1)
    mask = pws_label.sum(-1).sum(-1).view(-1)

    pws_hrs_mask = (mask).float()
    # pws_hrs_count = model_hrs_mask * pws_hrs_task_ids.view(-1)
    pws_hrs_loss = compute_weighted_loss(pws_hrs_loss, pws_hrs_mask)

    return pws_hrs_loss

def PairwiseClickLoss(x, label, list_length):
    pws_model_logits = pairwise_tensor(x.view(-1, list_length), list_length)
    pws_label = pairwise_tensor(label.view(-1, list_length), list_length)
    pws_label = torch.cat((torch.unsqueeze(pws_label[:, :, :, 0] > pws_label[:, :, :, 1], -1).float(),
                               torch.unsqueeze(pws_label[:, :, :, 0] < pws_label[:, :, :, 1], -1).float()), -1)
    pws_model_scores = torch.nn.functional.log_softmax(pws_model_logits,
                                                           -1)  # (BS/ListLength,ListLength,ListLength,2)

    loss = -(pws_model_scores * pws_label).sum(-1)  # (BS/ListLength,ListLength,ListLength)
    pws_click_loss = loss.sum(-1).view(-1)
    mask = pws_label.sum(-1).sum(-1).view(-1)

    pws_click_mask = mask.float()
    pws_click_count = pws_click_mask.view(-1)
    pws_click_loss = (pws_click_loss * pws_click_mask.view(-1)).sum() / (pws_click_count.sum() + 1e-7)

    return pws_click_loss