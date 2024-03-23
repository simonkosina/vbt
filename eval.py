"""
Evaluates the models on the test dataset,
creating Precision-Recall and ROC curves.
"""

import click
import os
import cv2
import glob
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import ast

from odt import run_odt
from tflite_runtime.interpreter import Interpreter
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score

LABEL = 'barbell'


class PythonLiteralOption(click.Option):
    """
    Custom option class for passing in a list
    of values as an argument using click.

    See:
    https://stackoverflow.com/a/47730333
    """

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


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


def create_detections_df(models, img_dir, annotations, export_path, num_threads):
    """
    Run inference on the images in img_dir using the provided
    models, assign these detection to proper ground truth bounding
    boxes defined in the annotations dict and export the created dataframe.
    """

    img_files = glob.glob(f'{img_dir}/*.jpg')
    detections = {}

    for m in models:
        interpreter = Interpreter(model_path=m, num_threads=num_threads)
        interpreter.allocate_tensors()

        model_detections = {}

        for f in img_files:
            img = cv2.imread(f)
            height, width, _ = img.shape

            results = run_odt(
                frame=img,
                interpreter=interpreter,
                threshold=0
            )

            for r in results:
                r['bounding_box'] = scaled_bbox(
                    r['bounding_box'], (1, 1), (height, width))

            model_detections[os.path.basename(f)] = results

        detections[os.path.basename(m).split('.')[0]] = model_detections

    scores = []
    models = []
    ious = []

    for file, gt_bboxes in annotations.items():
        for model, model_detections in detections.items():
            gt_idxs, det_idxs, det_ious = match_bboxes(
                gt_bboxes=gt_bboxes,
                det_bboxes=np.array(
                    list(map(lambda x: x['bounding_box'], model_detections[file])))
            )

            for i, det_idx in enumerate(det_idxs):
                scores.append(model_detections[file][det_idx]['score'])
                ious.append(det_ious[i])
                models.append(model)

    df_det = pd.DataFrame({
        'Score': scores,
        'Model': models,
        'IoU': ious
    })

    df_det.to_pickle(export_path)

    return df_det


def plot_precision_recall(df, gt_total, fig_dir, iou_threshold):
    """
    Based on the provided dataframe and total number of ground
    truth bounding boxes in the annotations, plot the precision-recall
    curve and save it to fig_dir.
    """

    df['TP'] = df['Label'].astype(int)
    df['FP'] = np.logical_not(df['Label']).astype(int)

    # Sort by score and calculate precision and recall values using
    # accumulated counts of true and false positives
    df.sort_values(by='Score', inplace=True, ascending=False)
    df['acc_tp'] = df.groupby('Model')['TP'].cumsum()
    df['acc_fp'] = df.groupby('Model')['FP'].cumsum()
    df['Precision'] = df['acc_tp'] / (df['acc_tp'] + df['acc_fp'])
    df['Recall'] = df['acc_tp'] / gt_total

    # Get average precision values for each model
    aps = {}
    for m in pd.unique(df["Model"]):
        dfm = df.query("Model == @m")
        scores = dfm["Score"]
        labels = dfm["Label"]

        aps[m] = average_precision_score(labels, scores)

    ax = sns.lineplot(data=df, x='Recall',
                      y='Precision', hue='Model',
                      errorbar=None)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    for i, model in enumerate(labels):
        labels[i] += f', AP={aps[model]:.4f}'

    ax.set_title(f'Precision-Recall curve, IoU threshold = {iou_threshold}')
    ax.legend(handles, labels, loc='lower left')

    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='major', color='gray',
            linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='gray',
            linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'precision_recall.pdf'))
    plt.close()


def plot_roc(df, fig_dir, iou_threshold, score_thresholds=None):
    """
    Based on the provided dataframe plot the ROC curve
    for each models and store the plot in fig_dir.

    If score thresholds list is provided find the closest
    threshold scores to these values on the ROC curve
    and create a plot for each model displaying the points
    on the ROC curve.
    """
    rocs = []
    roc_aucs = {}

    # Calculate false/true positive rates, thresholds and AUC scores for each model
    for m in pd.unique(df["Model"]):
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

    # Concatenate individual model dataframes
    df_roc = pd.concat(rocs, ignore_index=True)

    # Plot individual ROC curves with AUC scores
    ax = sns.lineplot(data=df_roc, x='FP Rate',
                      y='TP Rate', hue='Model',
                      errorbar=None)

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    for i, model in enumerate(labels):
        labels[i] += f', AUC={roc_aucs[model]:.4f}'

    ax.set_title(f'ROC curve, IoU threshold = {iou_threshold}')
    ax.legend(handles, labels, loc='lower right')

    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    ax.grid(which='major', color='gray',
            linestyle='-', linewidth=0.5, alpha=0.7)
    ax.grid(which='minor', color='gray',
            linestyle=':', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'roc.pdf'))
    plt.close()

    if score_thresholds is None or len(score_thresholds) == 0:
        return

    # For each model find the given thresholds on the
    # ROC curve and create separate plots
    for m in pd.unique(df["Model"]):
        for handle, label in zip(handles, labels):
            if label.startswith(m):
                model_color = handle.get_color()

        dfm = df_roc.query("Model == @m")

        ax = sns.lineplot(data=dfm, x='FP Rate',
                          y='TP Rate', hue='Model',
                          errorbar=None, palette=[model_color])

        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        _handles, _labels = ax.get_legend_handles_labels()
        for i, model in enumerate(_labels):
            _labels[i] += f', AUC={roc_aucs[model]:.4f}'

        ax.set_title(
            f'ROC curve with score thresholds, IoU threshold = {iou_threshold}')
        ax.legend(_handles, _labels, loc='lower right')

        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))

        ax.grid(which='major', color='gray',
                linestyle='-', linewidth=0.5, alpha=0.7)
        ax.grid(which='minor', color='gray',
                linestyle=':', linewidth=0.5, alpha=0.5)

        for i, v in enumerate(score_thresholds):
            diffs = abs(dfm['Threshold'] - v)
            closest_row = dfm.loc[diffs.idxmin()]

            fpr = closest_row["FP Rate"]
            tpr = closest_row["TP Rate"]
            threshold = closest_row["Threshold"]

            text = f'{threshold:.4f}'
            ax.annotate(text,
                        xy=(fpr, tpr),
                        xycoords='data',
                        xytext=((len(score_thresholds) - i)*8, - (i + 1) * 15),
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
        plt.savefig(os.path.join(fig_dir, f'roc_{m}_thresholds.pdf'))
        plt.close()


@click.command()
@click.argument('models', type=str, nargs=-1)
@click.option('--img_dir', default='data/test', help='Directory containing the JPG test images.', show_default=True)
@click.option('--annotations_dir', default='data/test', help='Directory containing the XML annotation files.', show_default=True)
@click.option('--fig_dir', default=None, help='Directory for saving the figures. If not set the figures won\'t be saved.', show_default=True)
@click.option('--iou_threshold', default=0.5, type=float, help='Intersection over union threshold to label detections as correct or not when calculated against the ground truth bounding boxes.', show_default=True)
@click.option('--threads', default=4, help='Number of threads to use for detection model inference.', show_default=True)
@click.option('--detections_df', default='dfs/eval_detections.pkl.gz', help='Path for storing/reading the detection results dataframe.', show_default=True)
@click.option('--replace_df', is_flag=True, help='If exists, replace the detections dataframe.', show_default=True)
@click.option('--score_thresholds', default='[]', cls=PythonLiteralOption, help='List of score thresholds to plot on the ROC curves, e.g. "[0.2, 0.5]".', show_default=True)
def main(models, img_dir, annotations_dir, fig_dir, iou_threshold, threads, detections_df, replace_df, score_thresholds):
    """
    Plot Precision-Recall and ROC curves for the specified models.
    """
    sns.set_theme(style='ticks')
    sns.set_palette('Set2')

    # Load ground truth bounding boxes from annotations
    annotation_files = glob.glob(f'{annotations_dir}/*.xml')
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

    if not os.path.exists(detections_df) or replace_df:
        print(f"Creating dataframe '{detections_df}'.")
        df = create_detections_df(
            models, img_dir, annotations, detections_df, threads)
    else:
        print(f"Loading dataframe '{detections_df}'.")
        df = pd.read_pickle(detections_df)

    # Total number of ground truth bounding boxes
    gt_total = sum([len(gt_bboxes) for gt_bboxes in annotations.values()])

    # Classify detections as correct or not based on the IoU threshold
    df['Label'] = df['IoU'] > iou_threshold

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)

        plot_precision_recall(df.copy(), gt_total, fig_dir, iou_threshold)
        plot_roc(df.copy(), fig_dir, iou_threshold, score_thresholds)


if __name__ == "__main__":
    main()
