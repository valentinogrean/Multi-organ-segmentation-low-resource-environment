import nibabel
import numpy as np
import os

def rename():
  for x in range(101,151):
    os.chdir('/home/vali/src/datasets/StructSeg_Thoracic_OAR/Patient_' + str(x).zfill(2))
    os.rename('data.nii.gz', 'imaging.nii.gz')
    os.rename('label.nii.gz', 'segmentation.nii.gz')

def inspect():
  for x in range(1,151):
    filename = '/home/vali/src/datasets/multi-organ-segthor-trachea-1/Patient_' + str(x).zfill(2) + '/imaging.nii.gz'
    if os.path.isfile(filename):
      segmentation = nibabel.load(filename)
      pred_array = np.asanyarray(segmentation.dataobj)
    
      print('Patient_' + str(x).zfill(2) + ': ' + str(segmentation.header['pixdim'][1]) + ':' + 
        str(segmentation.header['pixdim'][2]) + ':' + str(segmentation.header['pixdim'][3]))

def concatenate_evaluations():
  for x in range(41, 61):
    filename_final = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-22-0.97only-1.78-single-class-only-better-overlap-clipping-3-aggregated-evaluations-16-48-72-104/04-aorta/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction1 = nibabel.load(filename_final)
    pred1_array = np.asanyarray(prediction1.dataobj)
    ceva = np.unique(pred1_array)
    
    filename_organ = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-22-0.97only-1.78-single-class-only-better-overlap-clipping-3-aggregated-evaluations-16-48-72-104/72/04-aorta/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction2 = nibabel.load(filename_organ)
    pred2_array = np.asanyarray(prediction2.dataobj)
    ceva = np.unique(pred2_array)

    final_train_array = np.where(pred1_array == 0, pred2_array, pred1_array)
    
    corrected_img = nibabel.Nifti1Image(final_train_array, prediction1.affine, prediction1.header)
    nibabel.save(corrected_img, filename_final)

def concatenate_single_multi():
  for x in range(41, 61):
    filename_final = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-28-multi-class-72-48-all-segthor+best-single 23/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction1 = nibabel.load(filename_final)
    pred1_array = np.asanyarray(prediction1.dataobj)
    ceva = np.unique(pred1_array)
    
    filename_organ = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-28-multi-class-72-48-all-segthor+best-single 23/segthor-22/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction2 = nibabel.load(filename_organ)
    pred2_array = np.asanyarray(prediction2.dataobj)
    ceva = np.unique(pred2_array)

    final_train_array = np.where(pred1_array == 0, pred2_array, pred1_array)
    
    corrected_img = nibabel.Nifti1Image(final_train_array, prediction1.affine, prediction1.header)
    nibabel.save(corrected_img, filename_final)

def concatenate():
  for x in range(41, 61):
    filename_final = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-28-multi-class-72-48-all-segthor+best-single 23/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction1 = nibabel.load(filename_final)
    pred1_array = np.asanyarray(prediction1.dataobj)
    ceva = np.unique(pred1_array)
    
    filename_organ = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-28-multi-class-72-48-all-segthor+best-single 23/segthor-21/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction2 = nibabel.load(filename_organ)
    pred2_array = np.asanyarray(prediction2.dataobj)
    ceva = np.unique(pred2_array)

    #final_train_array = np.where(np.logical_or(train_array == 3, np.logical_or(train_array == 1, train_array == 2)),train_array, pred_array)
    final_train_array = np.where(np.logical_and(pred1_array == 0, pred2_array == 3), 3, pred1_array)
    
    corrected_img = nibabel.Nifti1Image(final_train_array, prediction1.affine, prediction1.header)
    nibabel.save(corrected_img, filename_final)

def update_organs():
  for x in range(41, 61):
    filename = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-28-multi-class-72-48-all-segthor+best-single 23/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction = nibabel.load(filename)
    array = np.asanyarray(prediction.dataobj)
    
    array[array == 3] = 0

    corrected_img = nibabel.Nifti1Image(array, prediction.affine, prediction.header)
    nibabel.save(corrected_img, filename)

def remove_organs():
  for x in range(101, 151):
    filename = '/home/vali/src/datasets/StructSeg_Thoracic_OAR_trachea/Patient_' + str(x).zfill(2) + '/segmentation.nii.gz'
    prediction = nibabel.load(filename)
    array = np.asanyarray(prediction.dataobj)
    
    array[array == 1] = 0
    array[array == 2] = 0
    array[array == 3] = 0
    array[array == 4] = 0
    array[array == 5] = 1
    array[array == 6] = 0

    corrected_img = nibabel.Nifti1Image(array, prediction.affine, prediction.header)
    nibabel.save(corrected_img, filename)

def prepare_organs():
  for x in range(1, 41):
    filename = '/home/vali/src/datasets/multi-organ-segthor-heart/Patient_' + str(x).zfill(2) + '/segmentation.nii.gz'
    prediction = nibabel.load(filename)
    array = np.asanyarray(prediction.dataobj)
    
    array[array == 1] = 0
    array[array == 2] = 1
    array[array == 3] = 0
    array[array == 4] = 0
    
    corrected_img = nibabel.Nifti1Image(array, prediction.affine, prediction.header)
    nibabel.save(corrected_img, filename)

def convert_to_int():
  file_path = '/home/vali/src/datasets/test/Patient_43.nii.gz'
  file = nibabel.load(file_path)

  ceva = file.header.get_data_dtype()
  altceva = np.unique(file.get_data())

  new_data = np.copy(np.asanyarray(file.dataobj))
  hd = file.header

  new_dtype = np.int32
  new_data = np.int32(np.clip(new_data + 0.5,0,5))
  altceva1 = np.unique(new_data)
  file.set_data_dtype(new_dtype)

  corrected_img = nibabel.Nifti1Image(new_data, file.affine, file.header)
  nibabel.save(corrected_img,file_path)

def correct_header():
  for x in range(43, 44):
    prediction_file = '/home/vali/src/NeuralNetworks/predictions/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction = nibabel.load(prediction_file)
    segmentation = nibabel.load('/home/vali/src/datasets/multi-organ-segthor/Patient_' + str(1).zfill(2) + '/imaging.nii.gz')

    corrected_img = nibabel.Nifti1Image(np.asanyarray(prediction.dataobj), segmentation.affine, segmentation.header)
    nibabel.save(corrected_img,prediction_file)


def remove_large_values():
  for x in range(40):
    filename = 'd:\\Doctorat\\Datasets\\multi-organ-segthor\\case_00' + str(x).zfill(2) + '\\imaging.nii.gz'

    prediction = nibabel.load(filename)
    array = prediction.get_fdata()
    array[array > 3000] = 3000

    corrected_img = nibabel.Nifti1Image(array, prediction.affine, prediction.header)
    nibabel.save(corrected_img, filename)


def find_min_max():
  max = -1000
  min = 1000
  for x in range(1, 1000):
    filename_image = '/home/vali/src/datasets/multi-organ-segthor-trachea-1/Patient_' + str(x).zfill(2) + '/imaging.nii.gz'
    if os.path.isfile(filename_image):
      image = nibabel.load(filename_image)
      pred_image_array = np.asanyarray(image.dataobj)
          
      filename_segmentation = '/home/vali/src/datasets/multi-organ-segthor-trachea-1/Patient_' + str(x).zfill(2) + '/segmentation.nii.gz'
      segmentation = nibabel.load(filename_segmentation)
      pred_segmentation_array = np.asanyarray(segmentation.dataobj)
      
      final_train_array = np.where(pred_segmentation_array == 1, pred_image_array, 0)

      ceva = np.unique(final_train_array)

      with open('your_file.txt', 'w') as f:
        for item in ceva:
          f.write("%s\n" % item)

      print(str(x) + ' max: ' + str(np.max(final_train_array)) + ' min: '+ str(np.min(final_train_array)))
      if np.max(final_train_array) > max :
        max = np.max(final_train_array)
      if np.min(final_train_array) < min :
        min = np.min(final_train_array)
  print(max)
  print(min)

def intersection():
  for x in range(41, 61):
    filename_final = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-25-intersection/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction1 = nibabel.load(filename_final)
    pred1_array = np.asanyarray(prediction1.dataobj)
    ceva = np.unique(pred1_array)
    
    filename_organ = '/home/vali/src/NeuralNetworks/predictions_saves/segthor-25-intersection/22/Patient_' + str(x).zfill(2) + '.nii.gz'
    prediction2 = nibabel.load(filename_organ)
    pred2_array = np.asanyarray(prediction2.dataobj)
    ceva = np.unique(pred2_array)

    #final_train_array = np.where(np.logical_or(train_array == 3, np.logical_or(train_array == 1, train_array == 2)),train_array, pred_array)
    final_train_array = np.where(pred1_array == pred2_array, pred1_array, 0)
    
    corrected_img = nibabel.Nifti1Image(final_train_array, prediction1.affine, prediction1.header)
    nibabel.save(corrected_img, filename_final)

concatenate_single_multi()