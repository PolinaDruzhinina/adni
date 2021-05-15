"""The project is inspired by the clinica /clinicadl library, the code is taken from https://github.com/aramis-lab/AD-DL"""

import torch
import numpy as np
import os
import pandas as pd
from time import time
import logging
import shutil
from sklearn.metrics import roc_auc_score
from model import load_model

def test(model, dataloader, use_cuda, criterion, mode="image", use_labels=True):
    """
    Computes the predictions and evaluation metrics.
    Args:
        model: (Module) CNN to be tested.
        dataloader: (DataLoader) wrapper of a dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        use_labels (bool): If True the true_label will be written in output DataFrame and metrics dict will be created.
    Returns
        (DataFrame) results of each input.
        (dict) ensemble of metrics + total loss on mode level.
    """
    model.eval()
    dataloader.dataset.eval()

    columns = ["participant_id", "session_id", "true_label", "predicted_label"]

    softmax = torch.nn.Softmax(dim=1)
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    total_kl_loss = 0
    total_time = 0
    tend = time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if use_cuda:
                inputs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                inputs, labels = data['image'], data['label']

            outputs = model(inputs)
            if use_labels:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                if mode == "image":
                    row = [[sub, data['session_id'][idx], labels[idx].item(), predicted[idx].item()]]
                else:
                    normalized_output = softmax(outputs)
                    row = [[sub, data['session_id'][idx], data['%s_id' % mode][idx].item(),
                            labels[idx].item(), predicted[idx].item(),
                            normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]]

                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

            del inputs, outputs, labels
            tend = time()
        results_df.reset_index(inplace=True, drop=True)

    if not use_labels:
        results_df = results_df.drop("true_label", axis=1)
        metrics_dict = None
    else:
        metrics_dict = evaluate_prediction(results_df.true_label.values.astype(int),
                                           results_df.predicted_label.values.astype(int))
        metrics_dict['total_loss'] = total_loss
        metrics_dict['total_kl_loss'] = total_kl_loss
    torch.cuda.empty_cache()

    return results_df, metrics_dict


def evaluate_prediction(y, y_pred):
    """
    Evaluates different metrics based on the list of true labels and predicted labels.
    Args:
        y: (list) true labels
        y_pred: (list) corresponding predictions
    Returns:
        (dict) ensemble of metrics
    """

    true_positive = np.sum((y_pred == 1) & (y == 1))
    true_negative = np.sum((y_pred == 0) & (y == 0))
    false_positive = np.sum((y_pred == 1) & (y == 0))
    false_negative = np.sum((y_pred == 0) & (y == 1))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)*100

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)*100
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)*100
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)*100
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)*100
    else:
        npv = 0.0

    roc_auc_sk = roc_auc_score(y, y_pred)* 100
    balanced_accuracy = ((sensitivity + specificity) / 2)

    results = {'accuracy': round(accuracy,3),
               'balanced_accuracy': round(balanced_accuracy,3),
               'sensitivity': round(sensitivity,3),
               'specificity': round(specificity,3),
               'ppv': round(ppv,3),
               'npv': round(npv,3),
               'roc_auc_sk': round(roc_auc_sk, 3)
               }

    return results

def save_checkpoint(state, accuracy_is_best, loss_is_best, checkpoint_dir, filename='checkpoint.pth.tar',
                    best_accuracy='best_balanced_accuracy', best_loss='best_loss'):

    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(state, os.path.join(checkpoint_dir, filename))
    if accuracy_is_best:
        best_accuracy_path = os.path.join(checkpoint_dir, best_accuracy)
        if not os.path.exists(best_accuracy_path):
            os.makedirs(best_accuracy_path)
        shutil.copyfile(os.path.join(checkpoint_dir, filename),  os.path.join(best_accuracy_path, 'model_best.pth.tar'))

    if loss_is_best:
        best_loss_path = os.path.join(checkpoint_dir, best_loss)
        os.makedirs(best_loss_path, exist_ok=True)
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(best_loss_path, 'model_best.pth.tar'))


def test_single_cnn(model, output_dir, data_loader, subset_name, split, criterion, mode, logger, selection_threshold,
                    gpu=False):
    for selection in ["best_balanced_accuracy", "best_loss"]:
        # load the best trained model during the training
        model, best_epoch = load_model(model, os.path.join(output_dir, 'fold-%i' % split, 'models', selection),
                                       gpu=gpu, filename='model_best.pth.tar')

        results_df, metrics = test(model, data_loader, gpu, criterion, mode)
        logger.info("%s level %s balanced accuracy is %f for model selected on %s"
                    % (mode, subset_name, metrics["balanced_accuracy"], selection))

        mode_level_to_tsvs(output_dir, results_df, metrics, split, selection, mode, dataset=subset_name)
        print('presoft_voting')
        # Soft voting
        if data_loader.dataset.elem_per_image > 1:
            print('soft_voting')
            soft_voting_to_tsvs(output_dir, split, logger=logger, selection=selection, mode=mode,
                                dataset=subset_name, selection_threshold=selection_threshold)

def mode_level_to_tsvs(output_dir, results_df, metrics, fold, selection, mode, dataset='train', cnn_index=None):
    """
    Writes the outputs of the test function in tsv files.
    Args:
        output_dir: (str) path to the output directory.
        results_df: (DataFrame) the individual results per patch.
        metrics: (dict or DataFrame) the performances obtained on a series of metrics.
        fold: (int) the fold for which the performances were obtained.
        selection: (str) the metrics on which the model was selected (best_acc, best_loss)
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        dataset: (str) the dataset on which the evaluation was performed.
        cnn_index: (int) provide the cnn_index only for a multi-cnn framework.
    """
    if cnn_index is None:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    else:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index,
                                       selection)

    os.makedirs(performance_dir, exist_ok=True)

    results_df.to_csv(os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode)), index=False,
                      sep='\t')

    if metrics is not None:
        metrics["%s_id" % mode] = cnn_index
        if isinstance(metrics, dict):
            pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                                                    index=False, sep='\t')
        elif isinstance(metrics, pd.DataFrame):
            metrics.to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                           index=False, sep='\t')
        else:
            raise ValueError("Bad type for metrics: %s. Must be dict or DataFrame." % type(metrics).__name__)


def soft_voting_to_tsvs(output_dir, fold, selection, mode, dataset='test', num_cnn=None,
                        selection_threshold=None, logger=None, use_labels=True):
    """
    Writes soft voting results in tsv files.
    Args:
        output_dir: (str) path to the output directory.
        fold: (int) Fold number of the cross-validation.
        selection: (str) criterion on which the model is selected (either best_loss or best_acc)
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        dataset: (str) name of the dataset for which the soft-voting is performed. If different from training or
            validation, the weights of soft voting will be computed on validation accuracies.
        num_cnn: (int) if given load the patch level results of a multi-CNN framework.
        selection_threshold: (float) all patches for which the classification accuracy is below the
            threshold is removed.
        logger: (logging object) writer to stdout and stderr
        use_labels: (bool) If True the labels are added to the final tsv
    """
    if logger is None:
        logger = logging

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn)
    validation_df = retrieve_sub_level_results(output_dir, fold, selection, mode, validation_dataset, num_cnn)

    performance_path = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    os.makedirs(performance_path, exist_ok=True)

    df_final, metrics = soft_voting(test_df, validation_df, mode, selection_threshold=selection_threshold,
                                    use_labels=use_labels)

    df_final.to_csv(os.path.join(os.path.join(performance_path, '%s_image_level_prediction.tsv' % dataset)),
                    index=False, sep='\t')
    if use_labels:
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_path, '%s_image_level_metrics.tsv' % dataset),
                                                index=False, sep='\t')
        logger.info("image level %s balanced accuracy is %f for model selected on %s"
                    % (dataset, metrics["balanced_accuracy"], selection))

def retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Retrieve performance_df for single or multi-CNN framework.
    If the results of the multi-CNN were not concatenated it will be done here."""
    result_tsv = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection,
                              '%s_%s_level_prediction.tsv' % (dataset, mode))
    if os.path.exists(result_tsv):
        performance_df = pd.read_csv(result_tsv, sep='\t')

    else:
        concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn)
        performance_df = pd.read_csv(result_tsv, sep='\t')

    return performance_df

def concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Concatenate the tsv files of a multi-CNN framework"""
    prediction_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    for cnn_index in range(num_cnn):
        cnn_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index)
        performance_dir = os.path.join(cnn_dir, selection)
        cnn_pred_path = os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode))
        cnn_metrics_path = os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode))

        cnn_pred_df = pd.read_csv(cnn_pred_path, sep='\t')
        prediction_df = pd.concat([prediction_df, cnn_pred_df])
        os.remove(cnn_pred_path)

        if os.path.exists(cnn_metrics_path):
            cnn_metrics_df = pd.read_csv(cnn_metrics_path, sep='\t')
            metrics_df = pd.concat([metrics_df, cnn_metrics_df])
            os.remove(cnn_metrics_path)

        # Clean unused files
        if len(os.listdir(performance_dir)) == 0:
            os.rmdir(performance_dir)
        if len(os.listdir(cnn_dir)) == 0:
            os.rmdir(cnn_dir)

    prediction_df.reset_index(drop=True, inplace=True)
    if len(metrics_df) == 0:
        metrics_df = None
    else:
        metrics_df.reset_index(drop=True, inplace=True)
    mode_level_to_tsvs(output_dir, prediction_df, metrics_df, fold, selection, mode, dataset)

def soft_voting(performance_df, validation_df, mode, selection_threshold=None, use_labels=True):
    """
    Computes soft voting based on the probabilities in performance_df. Weights are computed based on the accuracies
    of validation_df.
    ref: S. Raschka. Python Machine Learning., 2015
    Args:
        performance_df: (DataFrame) results on patch level of the set on which the combination is made.
        validation_df: (DataFrame) results on patch level of the set used to compute the weights.
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        selection_threshold: (float) if given, all patches for which the classification accuracy is below the
            threshold is removed.
    Returns:
        df_final (DataFrame) the results on the image level
        results (dict) the metrics on the image level
    """

    # Compute the sub-level accuracies on the validation set:
    validation_df["accurate_prediction"] = validation_df.apply(lambda x: check_prediction(x), axis=1)
    sub_level_accuracies = validation_df.groupby("%s_id" % mode)["accurate_prediction"].sum()
    if selection_threshold is not None:
        sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
    weight_series = sub_level_accuracies / sub_level_accuracies.sum()

    # Sort to allow weighted average computation
    performance_df.sort_values(['participant_id', 'session_id', '%s_id' % mode], inplace=True)
    weight_series.sort_index(inplace=True)

    # Soft majority vote
    if use_labels:
        columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    else:
        columns = ['participant_id', 'session_id', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for (subject, session), subject_df in performance_df.groupby(['participant_id', 'session_id']):
        proba0 = np.average(subject_df["proba0"], weights=weight_series)
        proba1 = np.average(subject_df["proba1"], weights=weight_series)
        proba_list = [proba0, proba1]
        y_hat = proba_list.index(max(proba_list))

        if use_labels:
            y = subject_df["true_label"].unique().item()
            row = [[subject, session, y, y_hat]]
        else:
            row = [[subject, session, y_hat]]
        row_df = pd.DataFrame(row, columns=columns)
        df_final = df_final.append(row_df)

    if use_labels:
        results = evaluate_prediction(df_final.true_label.values.astype(int),
                                      df_final.predicted_label.values.astype(int))
    else:
        results = None

    return df_final, results


def test_ae(decoder, dataloader, use_cuda, criterion):
    """
    Computes the total loss of a given autoencoder and dataset wrapped by DataLoader.
    Args:
        decoder: (Autoencoder) Autoencoder constructed from a CNN with the Autoencoder class.
        dataloader: (DataLoader) wrapper of the dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
    Returns:
        (float) total loss of the model
    """
    decoder.eval()
    dataloader.dataset.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        outputs = decoder(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()

        del inputs, outputs, loss

    return total_loss