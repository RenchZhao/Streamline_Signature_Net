import dipy
import argparse
import os
from glob import glob
from dipy.io.streamline import save_vtk, load_trk, load_tractogram

if __name__ == "__main__":
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="Change input to vtk file.")  
    parser.add_argument('-inputDir', help='input tractography dataset contains file(s) of fibers supported by dipy.')
    parser.add_argument('-outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-refDir', default=None, required=False, help='reference of fiber if ext is not trk.')
    parser.add_argument('-ref_rel_path', default=None, required=False, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('--ext', default='trk', required=False, help='original_tract_filetype')

    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir
    refDir = args.refDir
    ref_rel_path = args.ref_rel_path
    ext = args.ext

    for subject in os.listdir(inputDir):
        ref_subject_path = None
        in_subject_path = os.path.join(inputDir, subject)
        if os.path.isdir(in_subject_path):
            out_subject_path = os.path.join(outputDir, subject)
            if refDir is not None:
                ref_subject_path = os.path.join(refDir, subject)
                if ref_rel_path is not None:
                    ref_subject_path = os.path.join(ref_subject_path, ref_rel_path)
                else:
                    ref_subject_path = os.path.join(ref_subject_path, subject+'.nii.gz')
            if not os.path.exists(out_subject_path):
                os.makedirs(out_subject_path)
            trk_ls = glob(os.path.join(in_subject_path,'**','*.{}'.format(ext)),recursive=True)
            for i in range(len(trk_ls)):
                out_fname = os.path.basename(trk_ls[i]).split('.')[0] + '.vtk'
                out_fname = os.path.join(out_subject_path, out_fname)
                if not os.path.exists(out_fname):
                    try:
                        if ext=='trk':
                            sft = load_trk(trk_ls[i],'same')
                            save_vtk(sft, out_fname)
                        else:
                            if ref_subject_path is None:
                                raise RuntimeError('no reference of input tracts specified.')
                            sft = load_tractogram(trk_ls[i],ref_subject_path)
                            save_vtk(sft, out_fname)
                    except Exception as e:
                        print(trk_ls[i])
                        print(e)

