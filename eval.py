# %%
# TODO: Comments, header, formatting, update README.md, click arguments
# TODO: ROC curves

import os
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize

from odt import run_odt
from tflite_runtime.interpreter import Interpreter
from matplotlib.ticker import MultipleLocator

ANNOTATIONS_DIR = 'data/test'
IMG_DIR = 'data/test'
MODEL_DIR = 'models'
LABEL = 'barbell'

THRESHOLDS = np.arange(0.05, 1.05, 0.05)
IOU_THRESHOLD = 0.5
THREADS = 4

MODEL = 'efficientdet_lite0_whole'


def create_bbox(bbox_element):
    """
    Create a [ymin, xmin, ymax, xmax] array
    representing a bounding box from the
    'bndbox' XML element.
    """

    xmin = bbox_element.find('xmin').text
    ymin = bbox_element.find('ymin').text
    xmax = bbox_element.find('xmax').text
    ymax = bbox_element.find('ymax').text

    return np.array([ymin, xmin, ymax, xmax], dtype=int)


def scaled_bbox(bbox, src_dim, dst_dim):
    """
    Scale the bounding from src_dim (height, width)
    to dst_dim (height, width).
    """

    src_height, src_width = src_dim
    dst_height, dst_width = dst_dim

    height_factor = dst_height / float(src_height)
    width_factor = dst_width / float(src_width)

    scaled_bbox = bbox * np.array([height_factor, width_factor] * 2)

    return scaled_bbox.astype(int)


def calculate_iou(det_box, gt_box):
    """
    Calculate the IoU for two bounding boxes.
    """

    intersection_ymin = max(det_box[0], gt_box[0])
    intersection_xmin = max(det_box[1], gt_box[1])
    intersection_ymax = min(det_box[2], gt_box[2])
    intersection_xmax = min(det_box[3], gt_box[3])

    intersection_area = max(0, intersection_ymax - intersection_ymin) * \
        max(0, intersection_xmax - intersection_xmin)

    det_box_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])

    union_area = det_box_area + gt_box_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def match_bboxes(gt_bboxes, det_bboxes):
    """
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.

    Addapted from:
    https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4#file-bbox_iou_evaluation-py

    Parameters
    ----------
    gt_bboxes, det_bboxes : Nx4 and Mx4 np array of bboxes [ymin, xmin, ymax, xmax]. 
      The number of bboxes, N and M, need not be the same.

    Returns
    -------
    (idxs_gt, idxs_pred, ious)
        idxs_gt, idxs_pred : indices into gt and pred for matches
        ious : corresponding IoU value of each match
    """

    MAX_IOU = 1.0
    MIN_IOU = 0.0

    n_gt = gt_bboxes.shape[0]
    n_pred = det_bboxes.shape[0]

    # ground truths x predictions IoU matrix
    iou_matrix = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou_matrix[i, j] = calculate_iou(det_bboxes[j, :], gt_bboxes[i, :])

    if n_pred > n_gt:
        # there are more predictions than ground-truth - add rows
        diff = n_pred - n_gt
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((diff, n_pred), MIN_IOU)),
                                    axis=0)

    if n_gt > n_pred:
        # more ground-truth than predictions - add columns
        diff = n_gt - n_pred
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((n_gt, diff), MIN_IOU)),
                                    axis=1)

    # call the Hungarian matching
    idxs_gt, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_gt[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    # sel_valid = (ious_actual > iou_threshold)

    # return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid]
    return idx_gt_actual, idx_pred_actual, ious_actual


# %%
if __name__ == "__main__":
    annotation_files = glob.glob(f'{ANNOTATIONS_DIR}/*.xml')
    annotations = {}

    for f in annotation_files:
        tree = ET.parse(f)
        root = tree.getroot()

        key = root.find('filename').text
        objects = root.findall('object')
        bboxes = []

        for o in objects:
            if o.find('name').text != LABEL:
                continue
            bboxes.append(create_bbox(o.find('bndbox')))

        annotations[key] = np.array(bboxes)

    img_files = glob.glob(f'{IMG_DIR}/*.jpg')
    detections = {}

    model_paths = glob.glob(f'{MODEL_DIR}/*.tflite')

    for m in model_paths:
        interpreter = Interpreter(model_path=m, num_threads=THREADS)
        interpreter.allocate_tensors()

        model_detections = {}

        for f in img_files:
            img = cv2.imread(f)
            height, width, _ = img.shape

            results = run_odt(
                frame=img,
                interpreter=interpreter,
                threshold=THRESHOLDS[0]
            )

            for r in results:
                r['bounding_box'] = scaled_bbox(
                    r['bounding_box'], (1, 1), (height, width))

            model_detections[os.path.basename(f)] = results

        detections[os.path.basename(m).split('.')[0]] = model_detections

# %%
    scores = []
    labels = []
    models = []

    for file, gt_bboxes in annotations.items():
        for model, model_detections in detections.items():
            gt_idxs, det_idxs, ious = match_bboxes(
                gt_bboxes=gt_bboxes,
                det_bboxes=np.array(
                    list(map(lambda x: x['bounding_box'], model_detections[file])))
            )

            for i, det_idx in enumerate(det_idxs):
                scores.append(model_detections[file][det_idx]['score'])
                labels.append(ious[i] > IOU_THRESHOLD)
                models.append(model)

    scores = np.array(scores)
    tps = np.array(labels).astype(int)
    fps = np.logical_not(labels).astype(int)

    gt_total = sum([len(gt_bboxes) for gt_bboxes in annotations.values()])

# %%
    from sklearn.metrics import average_precision_score

    df = pd.DataFrame({
        'Score': scores,
        'Model': models,
        'Label': labels,
        'TP': tps,
        'FP': fps,
    })

    df.sort_values(by='Score', inplace=True, ascending=False)
    df['acc_tp'] = df.groupby('Model')['TP'].cumsum()
    df['acc_fp'] = df.groupby('Model')['FP'].cumsum()
    df['Precision'] = df['acc_tp'] / (df['acc_tp'] + df['acc_fp'])
    df['Recall'] = df['acc_tp'] / gt_total

    aps = {}

    for m in pd.unique(df["Model"]):
        dfm = df.query("Model == @m")
        scores = dfm["Score"]
        labels = dfm["Label"]

        aps[m] = average_precision_score(labels, scores)
        

# %%
    # TODO: Display AP in the legend
    # TODO: Mention the IoU threshold
    sns.set_theme(style='ticks')
    sns.set_palette('Set2')

    ax = sns.lineplot(data=df, x='Recall',
                 y='Precision', hue='Model',
                 errorbar=None)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    for i, model in enumerate(labels):
        labels[i] += f', AP={aps[model]:.4f}'

    ax.set_title(f'Precision-Recall curve, IoU threshold = {IOU_THRESHOLD}')
    ax.legend(handles, labels, loc='lower left')

    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

    # TODO: Display AUC in the legend
# %%
    from sklearn.metrics import roc_curve, roc_auc_score

    rocs = []
    roc_aucs = {}

    for m in pd.unique(df["Model"]):
        # scores, labels = df.query("Model == @m")["Score", "Label"]
        dfm = df.query("Model == @m")
        scores = dfm["Score"]
        labels = dfm["Label"]

        fpr, tpr, thresholds = roc_curve(labels, scores)

        df_model_roc = pd.DataFrame({
            'FP Rate': fpr,
            'TP Rate': tpr,
            'Threshold': thresholds,
            'Model': m
        })

        rocs.append(df_model_roc)

        roc_aucs[m] = roc_auc_score(labels, scores)

    df_roc = pd.concat(rocs, ignore_index=True)
# %%

    ax = sns.lineplot(data=df_roc, x='FP Rate',
                 y='TP Rate', hue='Model',
                 errorbar=None)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    for i, model in enumerate(labels):
        labels[i] += f', AUC={roc_aucs[model]:.4f}'

    ax.set_title(f'ROC curve, IoU threshold = {IOU_THRESHOLD}')
    ax.legend(handles, labels, loc='lower right')

    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

# %%
    # TODO: Display threshold options from the app (0.1, 0.2, 0.3, 0.4, 0.5)
    # on the ROC curve for efficientdet_lite0_whole model.

    dfm = df_roc.query("Model == @MODEL")

    ax = sns.lineplot(data=dfm, x='FP Rate',
                 y='TP Rate', hue='Model',
                 errorbar=None)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    for i, model in enumerate(labels):
        labels[i] += f', AUC={roc_aucs[model]:.4f}'

    ax.set_title(f'ROC curve with score thresholds, IoU threshold = {IOU_THRESHOLD}')
    ax.legend(handles, labels, loc='lower right')

    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    for i, v in enumerate(thresholds):
        diffs = abs(dfm['Threshold'] - v)
        closest_row = dfm.loc[diffs.idxmin()]

        fpr = closest_row["FP Rate"]
        tpr = closest_row["TP Rate"]
        threshold = closest_row["Threshold"]

        text = f'{threshold:.4f}'
        print(text)
        ax.annotate(text,
            xy=(fpr, tpr),
            xycoords='data',
            xytext=((len(thresholds) - i)*8, - (i + 1) * 15),
            textcoords='offset points',
            arrowprops=dict(
                arrowstyle="->",
                color='k',
                connectionstyle="arc3,rad=-0.1",
                relpos=(0, 1)
            ),
            fontsize=10
        )

    plt.tight_layout()
    plt.show()
