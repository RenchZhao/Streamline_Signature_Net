
import dipy
import argparse
import os
from glob import glob
from dipy.io.streamline import load_tck
from cmd_utils import run_command

from dipy.io.utils import get_reference_info
from PIL import Image


def visu_one_subject(subj_tck_path, ref_img_path, out_subject_path, out_prefix=None, log=None):
    cmd='''

    mrview {0} --tractography.load {1}/ATR_right.tck  -voxel 0,0,105 -plane 2 -capture.prefix ATR_right
    mrview {0} --tractography.load {1}/ATR_left.tck  -voxel 0,0,105 -plane 2 -capture.prefix ATR_left

    mrview {0} --tractography.load {1}/T_PREM_right.tck  -voxel 0,0,122 -plane 2 -capture.prefix T_PREM_right
    mrview {0} --tractography.load {1}/T_PREM_left.tck  -voxel 0,0,122 -plane 2 -capture.prefix T_PREM_left

    mrview {0} --tractography.load {1}/T_PREC_left.tck -voxel 0,164,0 -plane 1 -capture.prefix T_PREC_left 
    mrview {0} --tractography.load {1}/T_PREC_right.tck -voxel 0,166,0 -plane 1 -capture.prefix T_PREC_right 

    mrview {0} --tractography.load {1}/CG_left.tck -voxel 116,0,0 -plane 0 -capture.prefix CG_left 
    mrview {0} --tractography.load {1}/CG_right.tck -voxel 147,0,0 -plane 0 -capture.prefix CG_right 
    

    '''.format(
        ref_img_path,
        subj_tck_path,
    )
    cmd_line_ls=cmd.split('\n')
    cmd_line_ls = [x for x in cmd_line_ls if len(x.strip())>0]

    cmd_suffix=' -capture.folder {} -capture.grab -exit -focus False -noannotations'.format(out_subject_path)
    # i=0
    for line in cmd_line_ls:
        # print(i)
        # i+=1
        if out_prefix is not None:
            line=line+out_prefix
        line = line+cmd_suffix
        try:
            run_command(line, log=log, shell=True)
        except Exception as e:
            print(e)

def remove_annotations_and_flip(image_path, out_path):
    image = Image.open(image_path)

    # 获取图像的大小
    width, height = image.size
    # print(width, height)

        # 设置第一个矩形区域的尺寸（a 高，b 宽）
    a = 50  # 第一个矩形区域的高度
    b = 150  # 第一个矩形区域的宽度

    # 设置第二个矩形区域的尺寸（c 高）
    c = 30  # 第二个矩形区域的高度

    # 设置需要切除的矩形区域的宽度
    d = 100  # 左边矩形区域的宽度
    e = 100  # 右边矩形区域的宽度
    f = 10  # 上部矩形区域的宽度

    # 获取第一个矩形区域的坐标（左下角）
    left = 0
    upper = height - a  # 第一个矩形区域的上边界
    right = b
    lower = height  # 第一个矩形区域的下边界

    # 在这个区域内填充黑色
    image.paste((0, 0, 0), (left, upper, right, lower))

    # 获取第二个矩形区域的坐标（从图像底部开始）
    upper2 = height - c  # 第二个矩形区域的上边界
    lower2 = height  # 第二个矩形区域的下边界

    # 使用整个图像宽度填充第二个矩形区域（确保矩形宽度为图像宽度）
    image.paste((0, 0, 0), (0, upper2, width, lower2))

    # 切除左边宽为d的矩形区域和右边宽为e的矩形区域
    image = image.crop((d, 0, width - e, height))

    # 切除顶部宽为f的矩形区域
    image = image.crop((0, f, width - d - e, height))

    # 获取图像的大小
    width, height = image.size
    # 去除中间上面的坐标轴标志
    left=int(width/2)-1
    upper=0
    right=int(width/2)+11
    lower=16
    image.paste((0, 0, 0), (left, upper, right, lower))


    # 去除右边中间的灰度标志
    left=width-20
    upper=height-130
    right=width
    lower=height
    image.paste((0, 0, 0), (left, upper, right, lower))


    # 判断是否需要水平翻转-如果不是矢状面，mrview结果左右是反的
    if 'FPT' not in image_path.upper() and 'AF' not in image_path.upper() and 'PORT' not in image_path.upper() and 'CG' not in image_path.upper():
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    # 保存修改后的图片到指定路径
    image.save(out_path)   

if __name__ == "__main__":
    # -----------------
    # Parse arguments
    # -----------------
    parser = argparse.ArgumentParser(description="Visualize infered tck file.")  
    parser.add_argument('-tckDir', help='input tractography dataset contains file(s) of tck fibers.')
    parser.add_argument('-refImgDir', help='Reference img for mrview loading.')
    parser.add_argument('-outputDir', help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('-ref_rel_path', default=None, required=False, help='rel_path for refImg.')

    args = parser.parse_args()

    tckDir = args.tckDir
    refImgDir = args.refImgDir
    ref_rel_path = args.ref_rel_path
    outputDir = args.outputDir

    for subject in os.listdir(tckDir):
        in_subject_path = os.path.join(tckDir, subject)
        if os.path.isdir(in_subject_path):
            out_subject_path = os.path.join(outputDir, subject)
            if not os.path.exists(out_subject_path):
                os.makedirs(out_subject_path)
            if ref_rel_path is None:
                ref_img_path = os.path.join(refImgDir, subject,subject+'.nii.gz')
            else:
                ref_img_path = os.path.join(refImgDir, subject,ref_rel_path)
            visu_one_subject(in_subject_path, ref_img_path, out_subject_path)
            out_img_ls = os.listdir(out_subject_path)
            for out_img in out_img_ls:
                out_img = os.path.join(out_subject_path,out_img)
                if out_img.endswith('png'):
                    remove_annotations_and_flip(out_img, out_img.replace('0000',''))
                    run_command('rm {}'.format(out_img))
