import os 
import sys
import torch
import nibabel
import numpy as np
import SimpleITK as sitk


def load_origin_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
        zoomfactor:
    """
    img = nibabel.load(filename)
    pixelspacing = img.header.get_zooms()
    zoomfactor = list(pixelspacing)
    zoomfactor.reverse()
    # data = img.get_data()
    data = np.asanyarray(img.dataobj)
    data = data.transpose(2, 1, 0)
#     print(data.shape)

    return data, zoomfactor

SLICE = 16

def get_classwise_dice(predict, soft_y):
    """
    get dice scores for each class in predict and soft_y
    """
    tensor_dim = len(predict.size())
    num_class  = list(predict.size())[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.permute(0, 2, 3, 4, 1)
        predict = predict.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.permute(0, 2, 3, 1)
        predict = predict.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    soft_y  = torch.reshape(soft_y,  (-1, num_class))
    predict = torch.reshape(predict, (-1, num_class))

    y_vol = torch.sum(soft_y,  dim = 0)
    p_vol = torch.sum(predict, dim = 0)
    intersect = torch.sum(soft_y * predict, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score



SLICE = 16
CLASS_NUM = 2
if __name__ == "__main__":
    fold = str(sys.argv[1])
    output_prediction_dir = f"/media/disk1/jansen/code_rad/Dataset_Rad3/iteration_{fold}/gtv_test/uncut_scale_unproc/nnunet_output"
    groundtruth_mask_dir = f"/media/disk1/jansen/code_rad/Dataset_Rad3/iteration_{fold}/gtv_test/uncut_scale_unproc/label"
    for nifti_out_file in os.listdir(output_prediction_dir):
        if not nifti_out_file.endswith(".nii.gz"):
            continue
        nifti_out_path = os.path.join(output_prediction_dir,nifti_out_file)
        nifti_gt_path = os.path.join(groundtruth_mask_dir,nifti_out_file.replace('image','mask').strip('.gz'))
        if not os.path.isfile(nifti_gt_path):
            print(f'Missing Ground Truth Path Invalid:{nifti_gt_path.split("/")[-1]} for output:{nifti_out_file}')
            continue
        img_nifti_out, _ = load_origin_nifty_volume_as_array(nifti_out_path)
        img_nifti_gt,_ = load_origin_nifty_volume_as_array(nifti_gt_path)    
        # print(img_nifti_out.shape,img_nifti_gt.shape)
        # print(np.unique(img_nifti_out),np.unique(img_nifti_gt))
        
        for idx in range((len(img_nifti_gt)//SLICE + 1)):
            img_nifti_out_slice = img_nifti_out[idx*SLICE:(idx+1)*SLICE]
            img_nifti_out_gt_slice = img_nifti_gt[idx*SLICE:(idx+1)*SLICE]
            if len(np.unique(img_nifti_out_gt_slice)) != CLASS_NUM:
                continue 
            print(img_nifti_out_slice.shape,img_nifti_out_gt_slice.shape)