import dipy
import argparse
import os
from glob import glob
from dipy.io.streamline import save_tck, load_vtk


from dipy.io.utils import get_reference_info

# load vtk需要reference, 可以查看每个被试reference是不是一样。
if __name__ == "__main__":
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="Compute FiberMap of input vtk file.")  
    parser.add_argument('-inputDir', help='input tractography dataset contains file(s) of fibers supported by dipy.')
    parser.add_argument('-ImgDir', help='input tractography dataset contains file(s) of fibers supported by dipy.')
    parser.add_argument('-outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-ref_rel_path', default=None, required=False, help='The output directory should be a new empty directory. It will be created if needed.')

    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir
    ImgDir = args.ImgDir

    ref_rel_path = args.ref_rel_path

    for subject in os.listdir(inputDir):
        in_subject_path = os.path.join(inputDir, subject)
        if os.path.isdir(in_subject_path):
            out_subject_path = os.path.join(outputDir, subject)
            if not os.path.exists(out_subject_path):
                os.makedirs(out_subject_path)
            vtk_ls = glob(os.path.join(in_subject_path,'*.vtk'))
            for i in range(len(vtk_ls)):
                out_fname = os.path.basename(vtk_ls[i]).split('.')[0] + '.tck'
                out_fname = os.path.join(out_subject_path, out_fname)
                if not os.path.exists(out_fname):
                    try:
                        if ref_rel_path is None:
                            sft = load_vtk(vtk_ls[i],get_reference_info(os.path.join(ImgDir, subject,subject+'.nii.gz')))
                        else:
                            sft = load_vtk(vtk_ls[i],get_reference_info(os.path.join(ImgDir, subject,ref_rel_path)))
                        # out_fname = out_fname.replace('_infer','')
                        save_tck(sft, out_fname)
                    except Exception as e:
                        print(vtk_ls[i])
                        print(e)

