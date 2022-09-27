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
    tensor_dim = len(predict.shape)
    num_class  = list(predict.shape)[1]
    if(tensor_dim == 5):
        soft_y  = soft_y.transpose(0, 2, 3, 4, 1)
        predict = predict.transpose(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        soft_y  = soft_y.transpose(0, 2, 3, 1)
        predict = predict.transpose(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))

    soft_y  = np.reshape(soft_y,  (-1, num_class))
    predict = np.reshape(predict, (-1, num_class))

    y_vol = np.sum(soft_y,  axis = 0)
    p_vol = np.sum(predict, axis = 0)
    intersect = np.sum(soft_y * predict, axis = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score

def get_soft_label(input_tensor, num_class,device='cpu'):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*np.ones_like(input_tensor)
        # print(torch.unique(torch.squeeze(temp_prob).cpu()),'printing tensor',i)
        tensor_list.append(temp_prob)
    output_tensor = np.concatenate(tensor_list, axis = 1)
    # output_tensor = output_tensor.double()
    return output_tensor

def binary_dice(s, g, resize = False):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))

    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
    return dice


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
        dice = binary_dice(img_nifti_out,img_nifti_gt)
        continue
        
        # print(img_nifti_out.shape,img_nifti_gt.shape)
        # print(np.unique(img_nifti_out),np.unique(img_nifti_gt))
        soft_out_seq = []
        soft_label_seq = []
        for idx in range((len(img_nifti_gt)//SLICE + 1)):
            img_nifti_out_slice = img_nifti_out[idx*SLICE:(idx+1)*SLICE]
            img_nifti_gt_slice = img_nifti_gt[idx*SLICE:(idx+1)*SLICE]
            # if len(np.unique(img_nifti_gt_slice)) != CLASS_NUM:
            #     continue 
            print(img_nifti_out_slice.shape,img_nifti_gt_slice.shape)
            soft_out_seq.append(get_soft_label(img_nifti_out_slice,CLASS_NUM))
            soft_label_seq.append(get_soft_label(img_nifti_gt_slice,CLASS_NUM))
        
        soft_label_seq = np.concatenate(soft_label_seq,axis=2)
        soft_out_seq = np.concatenate(soft_out_seq,axis=2)

        soft_label_seq = np.expand_dims(soft_label_seq,axis=0)
        soft_out_seq = np.expand_dims(soft_out_seq,axis=0)
        
        gtv_dice = get_classwise_dice(soft_out_seq,soft_label_seq)
        for c in range(CLASS_NUM):
            print('class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice[c]))
        print('Done Next Image\n\n')