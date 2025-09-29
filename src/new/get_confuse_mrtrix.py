from sklearn.metrics import confusion_matrix
import h5py
import os
import pandas as pd
def cm2excel(out_path, cm, label_names):
    df = pd.DataFrame(cm,columns=label_names)
    df.insert(0, ' ', label_names)
    df.to_excel(out_path, index=False)

def main(data_path, fold, out_path, metric_name):
    val_res = h5py.File(os.path.join(data_path, str(fold), 'entire_data_validation_results_best_{}.h5'.format(metric_name)), "r")

    predicted_lst = val_res['val_predictions'][:]
    labels_lst = val_res['val_labels'][:]
    val_res.close()

    with h5py.File(os.path.join(data_path, str(fold), 'label_names.h5'),'r') as label_names_h5:
        label_names = label_names_h5['y_names'][:]
        label_names = [str(X) for X in label_names]
    #print(labels_lst, predicted_lst)
    cm = confusion_matrix(labels_lst, predicted_lst) # 横是标签，竖是预测
    del label_names[0]

    cm2excel(out_path, cm, label_names)

data_path = '/path/to/your/model_weight' 
fold = 2
metric_name = 'f1'
out_path = os.path.join(data_path, str(fold), 'confusion_matrix.xlsx')
main(data_path, fold, out_path, metric_name)
'''
for fold in range(1,6):
    out_path = os.path.join(data_path, str(fold), 'confusion_matrix.xlsx')
    main(data_path, fold, out_path, metric_name)'''


