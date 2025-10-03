import os

# input_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/imagesTr" #TEST Set DIr  #Change this to the path of the dataset
# input_dir ='/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset004_AVM_TOFMRA/imagesTr'
input_dir ='/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset004_AVM_TOFMRA/imagesVal_fold0'
# base_output_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset003_AVM_T1+C/" #Change this to where you want the segmentation to output
# base_output_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset004_AVM_TOFMRA/"
base_output_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset004_AVM_TOFMRA/"
os.makedirs(base_output_dir, exist_ok=True)
# base_model_path = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset003_AVM_T1+C/" # Change this to your model path, the .pth file!
base_model_path = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset004_AVM_TOFMRA/"
# base_model_path = '/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset004_AVM_TOFMRA/ViTTrainer_lr_1e_minus4__nnUNetResEncUNetLPlans__3d_fullres/fold_0'
# base_model_path = '/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset004_AVM_TOFMRA/ViTTrainer_lr_1e_minus4__nnUNetResEncUNetLPlans__3d_fullres/'#fold_i
# trainers = ["ViTTrainer_lr_1e_minus4"] # Change the trainer  , "nnUNetCLSTrainer", "nnUNetCLSTrainer_lr_1e_minus3", "DenseNetTrainer", "DenseNetTrainer_lr_1e_minus3", "SEResNetTrainer", "SEResNetTrainer_lr_1e_minus3",
trainers = ["SwinViTTrainer_ep300_NoMirroring"]
plans = ['nnUNetResEncUNetLPlans'] #  'nnUNetPlans', This is the current plans that are availabe in nnUNet # 
configs = ['3d_fullres']#['3d_fullres_nnssl_ckpt'] # Change to the configurations you want, 3d_fullres_BN_4BS
checkpoints = ["checkpoint_bestacc"]#["checkpoint_latest"] # [f"checkpoint_{i}" for i in range(800, 1000, 100)] + This can be changed to the checkpoints you do have
for i in range(5):
    input_dir = input_dir[:-1]+str(i)
    print (input_dir)
    for trainer in trainers:
        for plan in plans:
            for config in configs:
                for checkpoint in checkpoints:
                    # checkpoint_dir = os.path.join(base_output_dir, checkpoint)
                    # os.makedirs(checkpoint_dir, exist_ok=True)
                    output_dir = os.path.join(base_output_dir, f"{trainer}__{plan}__{config}/fold_{i}/", "cls_results/")
                    os.makedirs(output_dir, exist_ok=True)
                    model_path = os.path.join(base_model_path, f"{trainer}__{plan}__{config}/")
                    checkpoint_file = f"{checkpoint}.pth"
                    print(f"Trainer: {trainer}  Plan: {plan}   Config: {config}    Checkpoint: {checkpoint}")
                    cmd = f"python nnunet_cls_infer_nii.py --input_path {input_dir} -o {output_dir} --model_path {model_path} --fold {i} --checkpoint {checkpoint_file}"
                    print(f"Running: {cmd}")
                    os.system(cmd)

                    # cmd = f"python /hpf/projects/jquon/models/baseline_cls/nnUNet/nnunet_cls_infer_nii.py --pred {os.path.join(output_dir, 'results.csv')} --gt /bdm-das/ADSP_v1/PDCADxFoundation_n/val/val_cases.csv --out {os.path.join(output_dir, f'metrics.json')}"
                    # os.system(cmd)
