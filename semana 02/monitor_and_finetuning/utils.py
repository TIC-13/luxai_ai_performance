
import os
from datetime import datetime


def logging(message):
    print(f'{datetime.now().strftime("%Y-%M-%d %H:%M:%S")} {message}')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def get_current_time(format="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(format)


def get_metrics_from_class_scores(results, partition, classes):
        results_dict = dict()

        for idx in range(len(classes)):
            f1value = results[f"{partition}/F1-Scores"][idx]
            precisionvalue = results[f"{partition}/Precisions"][idx]
            recallvalue = results[f"{partition}/Recalls"][idx]
            disease_name = classes[idx]
            results_dict.update({
                f"{partition}_class_scores/{disease_name}_F1-Score":  f1value,
                f"{partition}_class_scores/{disease_name}_Precision": precisionvalue,
                f"{partition}_class_scores/{disease_name}_Recall":    recallvalue,
            })
        return results_dict


def getMeanValues(results, partition='', metrics=[], confMatrix=None):
    
    meanValues = list()

    for metric in metrics:
        
        if not confMatrix is None:
            metricValues = list()
            for clsId in range(confMatrix.shape[0]):
                if sum(confMatrix[:,clsId]) > 0: metricValues.append(results[f"{partition}/{metric}"][clsId])
        else:
            metricValues = results[f"{partition}/{metric}"]    
        
        print(metricValues)

        meanValues.append(sum(metricValues)/len(metricValues))
    
    return tuple(meanValues)
