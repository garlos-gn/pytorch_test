import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, jaccard_score
from scipy.spatial.distance import directed_hausdorff


def prepare_data(labels, predictions):
    lumen_gt = labels[:, :, :, 1]
    vessel_gt = labels[:, :, :, 2]
    lumen_p = predictions[:, :, :, 1]
    vessel_p = predictions[:, :, :, 2]
    return lumen_gt, vessel_gt, lumen_p, vessel_p


def measure(labels, predictions):
    lumen_gt, vessel_gt, lumen_p, vessel_p = prepare_data(labels, predictions)

    validation_labels = np.argmax(labels, axis=-1).astype('float32')
    predictions = np.argmax(predictions, axis=-1).astype('float32')

    print(classification_report(validation_labels.ravel(), predictions.ravel()))
    (precision, recall, fscore, support) = precision_recall_fscore_support(validation_labels.ravel(),
                                                                           predictions.ravel())
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('--------------------------------')

    auc_lumen = np.empty((lumen_p.shape[0],))
    jaccard_lumen = np.empty((lumen_p.shape[0],))
    dice_lumen = np.empty((lumen_p.shape[0],))
    hausdorff_lumen = np.empty((lumen_p.shape[0],))
    auc_vessel = np.empty((vessel_p.shape[0],))
    jaccard_vessel = np.empty((vessel_p.shape[0],))
    dice_vessel = np.empty((vessel_p.shape[0],))
    hausdorff_vessel = np.empty((vessel_p.shape[0],))

    n_imagenesl = 0

    for i in range(len(auc_lumen)):
        try:
            auc_lumen[i] = roc_auc_score(lumen_gt[i].ravel(), lumen_p[i].ravel())
            jaccard_lumen[i] = jaccard_score(lumen_gt[i].ravel(), lumen_p[i].ravel())
            dice_lumen[i] = (2 * jaccard_lumen[i]) / (jaccard_lumen[i] + 1)
            points1 = np.array(np.where(lumen_p[i, :] > 0)).transpose()
            points2 = np.array(np.where(lumen_gt[i, :] > 0)).transpose()
            hausdorff_lumen[i] = max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])
        except:
            n_imagenesl = n_imagenesl + 1
            continue

    n_imagenesv = 0

    for i in range(len(auc_vessel)):
        try:
            auc_vessel[i] = roc_auc_score(vessel_gt[i].ravel(), vessel_p[i].ravel())
            jaccard_vessel[i] = jaccard_score(vessel_gt[i].ravel(), vessel_p[i].ravel())
            dice_vessel[i] = (2 * jaccard_vessel[i]) / (jaccard_vessel[i] + 1)
            points1 = np.array(np.where(vessel_p[i, :] > 0)).transpose()
            points2 = np.array(np.where(vessel_gt[i, :] > 0)).transpose()
            hausdorff_vessel[i] = max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])
        except:
            n_imagenesv = n_imagenesv + 1
            continue

    print('Lumen: ')
    print()
    print('Not predicted Images: ' + str(n_imagenesl))
    print()
    print('--------------------------------')
    print()
    print('Mean AUC: ' + str(np.mean(auc_lumen)))
    print('Deviation AUC: ' + str(np.std(auc_lumen)))
    print()
    print('Mean Jaccard Index: ' + str(np.mean(jaccard_lumen)))
    print('Deviation Jaccard Index: ' + str(np.std(jaccard_lumen)))
    print()
    print('Mean Dice Index: ' + str(np.mean(dice_lumen)))
    print('Deviation Dice Index: ' + str(np.std(dice_lumen)))
    print()
    print('Hausdorff Distance: ' + str(np.mean(hausdorff_lumen)) + ' %')
    print('Hausdorff Deviation: ' + str(np.std(hausdorff_lumen)) + ' %')
    print()
    print('--------------------------------')
    print()
    print('Vessel: ')
    print()
    print('Not predicted Images: ' + str(n_imagenesv))
    print()
    print('--------------------------------')
    print()
    print('Mean AUC: ' + str(np.mean(auc_vessel)))
    print('Deviation AUC: ' + str(np.std(auc_vessel)))
    print()
    print('Mean Jaccard Index: ' + str(np.mean(jaccard_vessel)))
    print('Deviation Jaccard Index: ' + str(np.std(jaccard_vessel)))
    print()
    print('Mean Dice Index: ' + str(np.mean(dice_vessel)))
    print('Deviation Dice Index: ' + str(np.std(dice_vessel)))
    print()
    print('Hausdorff Distance: ' + str(np.mean(hausdorff_vessel)) + ' %')
    print('Hausdorff Deviation: ' + str(np.std(hausdorff_vessel)) + ' %')
    print()

    return auc_lumen, jaccard_lumen, dice_lumen, hausdorff_lumen, auc_vessel, jaccard_vessel, dice_vessel, \
           hausdorff_vessel