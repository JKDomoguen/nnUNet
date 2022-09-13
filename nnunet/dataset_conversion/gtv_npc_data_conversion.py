#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from nnunet.configuration import default_num_threads
from scipy.ndimage import label

# /media/disk1/jansen/code_rad/Dataset_Rad2/nnUNet_preprocessed
# /media/disk1/jansen/code_rad/Dataset_Rad2/nnUNet_trained_models

# export nnUNet_preprocessed="/media/disk1/jansen/code_rad/Dataset_Rad2/nnUNet_preprocessed"
# export RESULTS_FOLDER="/media/disk1/jansen/code_rad/Dataset_Rad2/nnUNet_trained_models"
def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    # train_dir = "/media/fabian/DeepLearningData/tmp/LITS-Challenge-Train-Data"
    # test_dir = "/media/fabian/My Book/datasets/LiTS/test_data"
    train_dir = "/media/disk1/jansen/code_rad/Dataset_Rad2/nifti_gtv_uni"
    # test_dir = "/media/fabian/My Book/datasets/LiTS/test_data"


    # output_folder = "/media/fabian/My Book/MedicalDecathlon/MedicalDecathlon_raw_splitted/Task029_LITS"
    output_folder = "/media/disk1/jansen/code_rad/Dataset_Rad2/nnUNet_raw_data_base/nnUNet_raw_data/Task510_Sample"
    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")

    # img_dir_te = join(output_folder, "imagesTs")

    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    # maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        print(data_file.split('/')[-1],seg_file.split('/')[-1])
        pat_id = data_file.split("/")[-1]
        pat_id = "train_" + pat_id.split("-")[-1][:-4]
        pat_id = pat_id.replace('_image','')

        img_itk = sitk.ReadImage(data_file)
        # img_array = sitk.GetArrayFromImage(img_itk)
        # print(img_array.shape,'original')
        # img_array = img_array.transpose(2, 1, 0)
        # img_array = np.expand_dims(img_array, axis = 0)
        # rev_img_itk = sitk.GetImageFromArray(img_array)
        # print(rev_img_itk.GetSize(),img_array.shape,'image')
        # sitk.WriteImage(rev_img_itk, join(img_dir, pat_id + "_0000.nii.gz"))
        sitk.WriteImage(img_itk, join(img_dir, pat_id + "_0000.nii.gz"))

        label_itk = sitk.ReadImage(seg_file)
        # label_array = sitk.GetArrayFromImage(label_itk)
        # print(label_array.shape,'original')
        # label_array = label_array.transpose(2, 1, 0)
        # label_array = np.expand_dims(label_array, axis = 0)
        # rev_label_itk = sitk.GetImageFromArray(label_array)
        # print(rev_img_itk.GetSize(),label_array.shape,'label')        
        # sitk.WriteImage(rev_label_itk, join(lab_dir, pat_id + ".nii.gz"))
        sitk.WriteImage(label_itk, join(lab_dir, pat_id + ".nii.gz"))

        return pat_id

    # def load_save_test(args):
    #     data_file = args
    #     pat_id = data_file.split("/")[-1]
    #     pat_id = "test_" + pat_id.split("-")[-1][:-4]

    #     img_itk = sitk.ReadImage(data_file)
    #     sitk.WriteImage(img_itk, join(img_dir_te, pat_id + "_0000.nii.gz"))
    #     return pat_id

    # nii_files_tr_data = subfiles(train_dir, True, "image", "nii", True)
    # nii_files_tr_seg = subfiles(train_dir, True, "mask", "nii", True)
    image_train_dir = os.path.join(train_dir,'image')
    mask_train_dir = os.path.join(train_dir,'mask')
    nii_files_tr_data = [os.path.join(image_train_dir,img_file) for img_file in os.listdir(image_train_dir) ]
    nii_files_tr_seg = [os.path.join(mask_train_dir,mask_file) for mask_file in os.listdir(mask_train_dir) ]

    print(nii_files_tr_data,'\n\n')
    print(nii_files_tr_seg)

    # nii_files_ts = subfiles(test_dir, True, "test-volume", "nii", True)

    p = Pool(default_num_threads)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    # test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "NPC-GTV"
    json_dict['description'] = "NN-UNet Segmentation of GTV in NPC"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "1.0"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "tumor",
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
    # json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)