import argparse
import os
from cmd_utils import run_command

from glob import glob
import pandas as pd

def stat_recognition_rate(out_excel_fname, input_dir, head=['bundle_name', 'reco_rate'], input_excel_prefix='recognition_num', reco_number_thresh=20):
    in_excel_ls = glob(os.path.join(input_dir,'**',input_excel_prefix+'*'),recursive=True)
    all_sub_num=len(in_excel_ls)
    reco_di = {}
    for i in range(all_sub_num):
        item = pd.read_excel(in_excel_ls[i]).values.tolist()
        for j in range(len(item)):
            if int(item[j][1])>=reco_number_thresh:
                reco_di[item[j][0]]=reco_di.get(item[j][0],0)+1
            else:
                reco_di[item[j][0]]=reco_di.get(item[j][0],0)
    df = list(reco_di.items())
    df = [[x[0], x[1]] for x in df]
    df = pd.DataFrame(df)
    df.columns = head
    df['reco_rate'] = df['reco_rate'].apply(lambda x: '{:.5f}'.format(float(x)/all_sub_num))
    
    df.to_excel(out_excel_fname, index=False)

if __name__=='__main__':
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="Compute FiberMap of input vtk file.")
    parser.add_argument('-reco_thresh', default=0.5, required=False, type=float, help='threshold of model argmax output bundle class that can be recognized.')
    parser.add_argument('-inputDir', help='input tractography dataset contains file(s) of fibers supported by dipy.')
    parser.add_argument('-refDir', help='reference of visualize fiber.')
    parser.add_argument('-ref_rel_path', default=None, required=False, help='The relative path of reference file below subject dir.')
    parser.add_argument('-model_name', help='Model type for inference.')
    parser.add_argument('-model_weight_path', help='Saved model weight for inference.')
    parser.add_argument('-tmpVtkDir', default=os.path.dirname(__file__)+'/tmpVtkDir', required=False, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-tmpResultDir', default=os.path.dirname(__file__)+'/tmpResultDir', required=False, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-resultTckDir', default=os.path.dirname(__file__)+'/resultTckDir', required=False, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-resultImgDir', default=os.path.dirname(__file__)+'/resultImgDir', required=False, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('--MRtrix3_tck', default=False, action='store_true', required=False, help='input tracts are in MRtrix3 .tck format.')
    parser.add_argument('--ext', default='tck', required=False, help='original_tract_filetype')
    parser.add_argument('--rm_tmp', default=False, action='store_true', required=False, help='remove tmp files')

    args = parser.parse_args()
    reco_thresh = args.reco_thresh
    inputDir = args.inputDir
    refDir = args.refDir
    ref_rel_path = args.ref_rel_path
    tmpVtkDir = args.tmpVtkDir
    tmpResultDir = args.tmpResultDir
    resultTckDir = args.resultTckDir
    resultImgDir = args.resultImgDir
    MRtrix3_tck = args.MRtrix3_tck
    ext = args.ext

    model_name = args.model_name
    model_weight_path = args.model_weight_path
    rm_tmp = args.rm_tmp
    
    if MRtrix3_tck:
        ext='tck'

    if not os.path.exists(tmpVtkDir):
        os.makedirs(tmpVtkDir)
    
    if not os.path.exists(tmpResultDir):
        os.makedirs(tmpResultDir)
    else:
        cmd = 'rm -r {}'.format(tmpResultDir)
        run_command(cmd)
    
    if not os.path.exists(resultTckDir):
        os.makedirs(resultTckDir)
    if not os.path.exists(resultImgDir):
        os.makedirs(resultImgDir)
    
    # tract convert to vtk
    subject_ls = []
    for subject in os.listdir(inputDir):
        in_subject_path = os.path.join(inputDir, subject)
        if os.path.isdir(in_subject_path):
            subject_ls.append(subject)
            if MRtrix3_tck:
                subjTmpVtkDir = os.path.join(tmpVtkDir, subject)
                if not os.path.exists(subjTmpVtkDir):
                    os.makedirs(subjTmpVtkDir)
                # print(os.path.join(in_subject_path,'**','*.{}'.format(ext)))
                trk_ls = glob(os.path.join(in_subject_path,'**','*.{}'.format(ext)),recursive=True)
                # print(trk_ls)
                for i in range(len(trk_ls)):
                    subjTmpVtkFname = os.path.basename(trk_ls[i]).split('.')[0] + '.vtk'
                    out_fname = os.path.join(subjTmpVtkDir, subjTmpVtkFname)
                    if not os.path.exists(out_fname):
                        cmd='tckconvert {} {} -binary'.format(trk_ls[i], out_fname)
                        run_command(cmd)
            else:
                cmd = 'python trk2vtk_dataset.py -inputDir {} -outputDir {} --ext {}'.format(inputDir, tmpVtkDir, ext)
                if refDir is not None:
                    cmd = cmd+' -refDir {}'.format(refDir)
                if ref_rel_path is not None:
                    cmd = cmd+' -ref_rel_path {}'.format(ref_rel_path)
                run_command(cmd)
    
    # inference
    for i in range(len(subject_ls)):
        in_subjVtk_path = os.path.join(tmpVtkDir, subject)
        in_vtk_ls = glob(os.path.join(in_subjVtk_path,'**','*.vtk'),recursive=True)
        subjResultImgDir = os.path.join(resultImgDir, subject)
        if not os.path.exists(subjResultImgDir):
            os.makedirs(subjResultImgDir)
        if len(in_vtk_ls)==1:
            in_vtk=in_vtk_ls[0]
            out_result_vtk=os.path.join(tmpResultDir, subject)
            out_result_excel = os.path.join(resultImgDir, subject, 'recognition_num.xlsx')
            cmd='python inference.py -reco_thresh {} -input_vtk {} -outputDir {} -model_name {} -model_weight_path {} -result_excel_fname {}'.format(reco_thresh, 
                                                                                                                                                    in_vtk, 
                                                                                                                                                    out_result_vtk, 
                                                                                                                                                    model_name, 
                                                                                                                                                    model_weight_path,
                                                                                                                                                    out_result_excel)
            if MRtrix3_tck:
                cmd = cmd + ' --RAS2LPS'
            if model_name=='DeepWMA':
                cmd = cmd + ' --RAS_3D'
            run_command(cmd)

        else:
            if len(in_vtk_ls)==0:
                print('subject: {} has no input tracts'.format(subject))
            for j in range(len(in_vtk_ls)):
                suffix = os.path.basename(in_vtk_ls[j]).split('.')[0]
                out_result_vtk=os.path.join(tmpResultDir, subject+'_'+suffix)
                out_result_excel = os.path.join(resultImgDir, subject, f'recognition_num_{suffix}.xlsx')
                in_vtk=in_vtk_ls[j]
                cmd='python inference.py -reco_thresh {} -input_vtk {} -outputDir {} -model_name {} -model_weight_path {} -result_excel_fname {}'.format(reco_thresh, 
                                                                                                                                                         in_vtk, 
                                                                                                                                                         out_result_vtk, 
                                                                                                                                                         model_name, 
                                                                                                                                                         model_weight_path,
                                                                                                                                                         out_result_excel)
                if MRtrix3_tck:
                    cmd = cmd + ' --RAS2LPS'
                if model_name=='DeepWMA':
                    cmd = cmd + ' --RAS_3D'
                run_command(cmd)
    
    # vtk result convert to tck
    cmd='python vtk2tck_dataset.py -inputDir {} -outputDir {} -ImgDir {}'.format(tmpResultDir, resultTckDir, refDir)
    if  ref_rel_path is not None:
        cmd=cmd+' -ref_rel_path {}'.format(ref_rel_path)
    run_command(cmd)

    # visu tracts
    cmd='python visu_tracts.py -tckDir {} -outputDir {} -refImgDir {}'.format(resultTckDir, resultImgDir, refDir)
    if  ref_rel_path is not None:
        cmd=cmd+' -ref_rel_path {}'.format(ref_rel_path)
    run_command(cmd)
    stat_recognition_rate(os.path.join(resultImgDir, 'recognition_rate.xlsx'), resultImgDir, reco_number_thresh=20)

    # remove tmp
    if rm_tmp:
        cmd = 'rm -r {}'.format(tmpVtkDir)
        run_command(cmd)
        cmd = 'rm -r {}'.format(tmpResultDir)
        run_command(cmd)
