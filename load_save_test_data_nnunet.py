import os
import sys
from collections import OrderedDict
import SimpleITK as sitk



def load_save_train(data_path,out_dir):
    pat_id = data_path.split("/")[-1]
    pat_id = pat_id.split(".")[0]

    img_itk = sitk.ReadImage(data_path)
    sitk.WriteImage(img_itk, os.path.join(out_dir, pat_id + "_0000.nii.gz"))

    return pat_id

if __name__ == '__main__':
    fold = str(sys.argv[1])
    ROOT_DIR = f"/media/disk1/jansen/code_rad/Dataset_Rad3/iteration_{fold}/gtv_test/uncut_scale_unproc"
    data_dir = os.path.join(ROOT_DIR,'data')
    data_nii_gz_dir = os.path.join(ROOT_DIR,'data_nii_gz')
    os.makedirs(data_nii_gz_dir,exist_ok=True)
    for data_file in os.listdir(data_dir):
        data_path = os.path.join(data_dir,data_file)
        load_save_train(data_path,data_nii_gz_dir)
        