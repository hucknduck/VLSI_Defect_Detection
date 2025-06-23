import torch

# IoU calculation between two sets of boxes (format: [x1,y1,x2,y2])
def box_iou(boxes1, boxes2):
    """
    Compute IoU between each pair of boxes in boxes1 and boxes2.
    boxes1: (N,4), boxes2: (M,4). Returns IoU matrix (N,M).
    """
    # Expand dims for broadcasting
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    lt = torch.max(
        boxes1[:, None, :2],  # (N,1,2)
        boxes2[:, :2][None, :, :]  # (1,M,2)
    )  # (N,M,2) top-left corner of overlap
    rb = torch.min(
        boxes1[:, None, 2:],  # (N,1,2)
        boxes2[:, 2:][None, :, :]  # (1,M,2)
    )  # (N,M,2) bottom-right corner of overlap

    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N,M)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union  # (N,M)


def encode_boxes(anchors, gt_boxes):
    """
    anchors: (N,4), gt_boxes: (N,4) matched ground-truth boxes for each anchor.
    Returns deltas (dx,dy,dw,dh) for each anchor.
    """
    widths_a  = anchors[:, 2] - anchors[:, 0]
    heights_a = anchors[:, 3] - anchors[:, 1]
    ctr_x_a   = anchors[:, 0] + 0.5 * widths_a
    ctr_y_a   = anchors[:, 1] + 0.5 * heights_a

    widths_gt  = gt_boxes[:, 2] - gt_boxes[:, 0]
    heights_gt = gt_boxes[:, 3] - gt_boxes[:, 1]
    ctr_x_gt   = gt_boxes[:, 0] + 0.5 * widths_gt
    ctr_y_gt   = gt_boxes[:, 1] + 0.5 * heights_gt

    dx = (ctr_x_gt - ctr_x_a) / widths_a
    dy = (ctr_y_gt - ctr_y_a) / heights_a
    dw = torch.log(widths_gt / widths_a)
    dh = torch.log(heights_gt / heights_a)

    deltas = torch.stack((dx, dy, dw, dh), dim=1)
    return deltas

def decode_boxes(anchors, deltas):
    """
    anchors: (N,4), deltas: (N,4) => refined boxes (N,4).
    """
    widths  = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x   = anchors[:, 0] + 0.5 * widths
    ctr_y   = anchors[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros_like(deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
    return pred_boxes
