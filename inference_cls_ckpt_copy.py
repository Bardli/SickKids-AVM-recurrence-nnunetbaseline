import os
input_dir ='/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset044_AVM_TOFMRA/imagesVal_fold0'
base_output_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/"
os.makedirs(base_output_dir, exist_ok=True)
base_model_path = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results/Dataset044_AVM_TOFMRA/"
trainers = ["DenseNetTrainer_ep300_NoMirroring","SEResNetTrainer_ep300_NoMirroring","SwinViTTrainer_ep300_NoMirroring","ViTTrainer_ep300_NoMirroring"] #["DenseNetTrainer_ep300_NoMirroring"] #["SEResNetTrainer_ep300_NoMirroring"] # trainers = ["SwinViTTrainer_ep300_NoMirroring"]#["ViTTrainer_lr_1e_minus4"] # Change the trainer  , "nnUNetCLSTrainer", "nnUNetCLSTrainer_lr_1e_minus3", "DenseNetTrainer", "DenseNetTrainer_lr_1e_minus3", "SEResNetTrainer", "SEResNetTrainer_lr_1e_minus3",
plans = ['nnUNetResEncUNetLPlans'] #  'nnUNetPlans', This is the current plans that are availabe in nnUNet # 
configs = ['3d_fullres']#['3d_fullres_nnssl_ckpt'] # Change to the configurations you want, 3d_fullres_BN_4BS
checkpoints = ["checkpoint_every20_best_balacc"]#,"checkpoint_bestauc"] #["checkpoint_every20_best_balacc", "checkpoint_bestauc","checkpoint_bestacc","checkpoint_final"]#["checkpoint_latest"] # [f"checkpoint_{i}" for i in range(800, 1000, 100)] + This can be changed to the checkpoints you do have
for i in range(6,11):
    input_dir = input_dir[:-1]+str(i)
    print (input_dir)
    for trainer in trainers:
        for plan in plans:
            for config in configs:
                for checkpoint in checkpoints:
                    # checkpoint_dir = os.path.join(base_output_dir, checkpoint)
                    # os.makedirs(checkpoint_dir, exist_ok=True)
                    output_dir = os.path.join(base_output_dir, f"{trainer}__{plan}__{config}/fold_{i}/", f"cls_results_{checkpoint}/") # "cls_results/")
                    os.makedirs(output_dir, exist_ok=True)
                    model_path = os.path.join(base_model_path, f"{trainer}__{plan}__{config}/")
                    checkpoint_file = f"{checkpoint}.pth"
                    print(f"Trainer: {trainer}  Plan: {plan}   Config: {config}    Checkpoint: {checkpoint}")
                    cmd = f"python nnunet_cls_infer_nii.py -i {input_dir} -o {output_dir} --model_path {model_path} --fold {i} --checkpoint {checkpoint_file}"
                    print(f"Running: {cmd}")
                    os.system(cmd)

                    # cmd = f"python /hpf/projects/jquon/models/nnunetcls_baseline/nnunet_cls_infer_nii.py --pred {os.path.join(output_dir, 'results.csv')} --gt /bdm-das/ADSP_v1/PDCADxFoundation_n/val/val_cases.csv --out {os.path.join(output_dir, f'metrics.json')}"
                    # cmd = f"python /hpf/projects/jquon/models/nnunetcls_baseline/nnunet_cls_infer_nii.py --pred {os.path.join(output_dir, 'results.csv')} --out {os.path.join(output_dir, f'metrics.json')}"
                    # os.system(cmd)

