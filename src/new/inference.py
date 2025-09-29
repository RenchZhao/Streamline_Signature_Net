# trk or vtk input. model infer. assign infered fiber together, not exceed recog_thresh success, else fail
# calculate every fiber number each classes



import argparse
import sys
sys.path.append('../')

import whitematteranalysis as wma
import numpy as np
import pandas as pd
import torch
import time
import os

from new.gen_train_h5 import label_di
from utils.dataset import InferVtkDataset, _feat_to_3D
from utils.logger import create_logger
from utils.model import get_model, load_model_weights
import torch.nn.functional as F
# from train_s1 import get_free_gpu

def RAS2LPS_and_feat_to_3D(points):
    return _feat_to_3D(RAS2LPS_transform(points))

def RAS2LPS_transform(points):
    assert points.shape[2]==3, 'Streamline input should be [B,N,3] format.'
    points[:,:,0] = -points[:,:,0]
    points[:,:,1] = -points[:,:,1]
    return points

def deposit_streamlines(streamline_di, label_names, data_lst, test_predicted_lst):
    for i in range(len(test_predicted_lst)):
        if test_predicted_lst[i]==0:
            continue
        streamline_di[label_names[test_predicted_lst[i] - 1]].append(data_lst[i])
    return streamline_di

def streamlines_di_save_vtk(streamline_di, label_names, out_dir):
    for i in range(len(label_names)):
        # out_vtk_fname = os.path.join(out_dir, label_names[i]+'_infer.vtk')
        out_vtk_fname = os.path.join(out_dir, label_names[i]+'.vtk')
        feat = np.array(streamline_di[label_names[i]])
        # print(feat.shape)
        if len(feat.shape)<3:
            # no fiber identified
            continue
        feat2vtk(feat, out_vtk_fname)

def stat_recognition_result(excel_fname, streamline_di, label_names, head=['bundle_name', 'recognized_num'], force=True):
    if force or not os.path.exists(excel_fname):
        df = []
        for i in range(len(label_names)):
            recognized_num = len(streamline_di[label_names[i]])
            df.append([label_names[i], recognized_num])
        df = pd.DataFrame(df)
        df.columns=head
        df.to_excel(excel_fname, index=False)
    else:
        print('recognition result file:{} exists, skip generation.'.format(excel_fname))

def model_inference(model, test_data_loader, streamline_di, label_names, script_name, logger, device, thresh, data_transform=None, result_transform='softmax'):
    """perform predition of model"""
    logger.info('')
    logger.info('===================================')
    logger.info('')
    logger.info('{} Start multi-cluster prediction.'.format(script_name))
    # Load model
    start_time = time.time()
    test_predicted_lst = []
    data_lst = []

    # trick: exp for data is equal to log for thresh
    if result_transform=='exp':
        thresh = torch.log(torch.tensor(thresh))

    with torch.no_grad():
        for j, data in (enumerate(test_data_loader, 0)):
            points = data
            if data_transform:
                points = torch.from_numpy(data_transform(points.numpy()).astype(np.float32))
            points = points.to(device)
            model = model.eval()    
            # predict
            pred = model(points)
            if not isinstance(pred, torch.Tensor):
                pred = pred[0]
            if result_transform=='softmax':
                pred = F.softmax(pred)

            prob, pred_idx = torch.max(pred, dim=1)
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            prob = prob.cpu().detach().numpy().tolist()
            
            # threshold flitering
            for i in range(len(pred_idx)):
                x=pred_idx[i]
                pred_idx[i]= x if prob[i]>thresh else 0
            test_predicted_lst.extend(pred_idx)
            data_lst.extend(data.numpy())

    end_time = time.time()
    pred_time = end_time - start_time
    logger.info('The total time of prediction is:{} s'.format(round((pred_time), 4)))
    logger.info('The test sample size is: {}'.format(len(test_predicted_lst)))
    streamline_di = deposit_streamlines(streamline_di, label_names, data_lst, test_predicted_lst)
    return test_predicted_lst, pred_time, streamline_di

def feat2vtk(features, out_vtk):
    number_of_fibers = features.shape[0]
    points_per_fiber = features.shape[1]
    tract = wma.fibers.FiberArray()
    # allocate array number of lines by line length
    tract.number_of_fibers = number_of_fibers #important to init
    tract.points_per_fiber = points_per_fiber #important to init
    tract.fiber_array_r = np.zeros((number_of_fibers,points_per_fiber))
    tract.fiber_array_a = np.zeros((number_of_fibers, points_per_fiber))
    tract.fiber_array_s = np.zeros((number_of_fibers,points_per_fiber))

    for lidx in range(0, number_of_fibers):
        for pidx in range(points_per_fiber):
            tract.fiber_array_r[lidx, pidx] = features[lidx, pidx, 0]
            tract.fiber_array_a[lidx, pidx] = features[lidx, pidx, 1]
            tract.fiber_array_s[lidx, pidx] = features[lidx, pidx, 2]
    
    out_pd_tract = tract.convert_to_polydata()
    wma.io.write_polydata(out_pd_tract, out_vtk)

if __name__=='__main__':
    script_name = '<inference>'
    label_names = sorted(list(label_di.items()), key=lambda x:x[1])
    label_names = [x[0] for x in label_names]

    streamline_di = {}
    for i in range(len(label_names)):
        streamline_di[label_names[i]]=[]
    
    # main
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="model infer for vtk file input.")  
    parser.add_argument('-input_vtk', help='vtk file for inference')
    parser.add_argument('-outputDir', help='The output directory for model inference .')
    parser.add_argument('-model_name', help='Model type for inference.')
    parser.add_argument('-model_weight_path', help='Saved model weight for inference.')
    parser.add_argument('-reco_thresh', default=0.5, required=False, type=float, help='threshold of model argmax output bundle class that can be recognized.')
    parser.add_argument('-result_excel_fname', default=None, required=False, type=str, help='The output directory for model inference.')
    parser.add_argument('--RAS2LPS', default=False, required=False, action='store_true', help='Transform for mrtrix3 vtk input.')
    parser.add_argument('--RAS_3D', default=False, required=False, action='store_true', help='Transform for DeepWMA.')
    parser.add_argument('--stream', type=int, default=15, required=False, help='stream size of data.')
    

    args = parser.parse_args()

    input_vtk = args.input_vtk
    out_path = args.outputDir
    model_name = args.model_name
    model_weight_path = args.model_weight_path
    RAS2LPS = args.RAS2LPS
    RAS_3D = args.RAS_3D
    result_excel_fname = args.result_excel_fname
    recoginition_thresh =args.reco_thresh

    # input_vtk = '/media/rench/MyBook/HCP/test/AF_left.vtk'
    # out_path = '/media/rench/MyBook/HCP/test/model_infer'
    # model_name='PointNetAndRawSig'
    # model_weight_path = '/media/rench/MyBook/HCP/test/best_f1_model.pth'
    #transform = _feat_to_3D

    # transform
    transform = None
    if RAS2LPS:
        if RAS_3D:
            transform = RAS2LPS_and_feat_to_3D
        else:
            transform = RAS2LPS_transform
    else:
        if RAS_3D:
            transform = _feat_to_3D

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    logger = create_logger(out_path)
    device = 'cpu' #get_free_gpu
    test_dataset = InferVtkDataset(input_vtk)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024,
        shuffle=False)
    model = load_model_weights(model_weight_path, model_name=model_name, num_classes=len(label_names)+1, stream=args.stream)
    model.to(device)

    result_transform='exp' # log_softmax back to softmax
    test_predicted_lst, pred_time, streamline_di = model_inference(model, test_data_loader, streamline_di, label_names, script_name, logger, device, thresh=recoginition_thresh, data_transform=transform, result_transform=result_transform)
    # print(streamline_di)
    if result_excel_fname is not None:
        stat_recognition_result(result_excel_fname, streamline_di, label_names)
    streamlines_di_save_vtk(streamline_di, label_names, out_dir=out_path)

