Pytorch implementation of NASA 
(https://virtualhumans.mpi-inf.mpg.de/nasa)

#Requirements:

1. Pytorch SMPL: https://github.com/chaitanya100100/TailorNet
or
   https://github.com/gulvarol/smplpytorch
2. Dependencies: nasa.yml

Path: to be edited

#Data generation

python data/prepare_data.py -frame <subject_id> -pose_file <pose_path_pkl> -beta_file <betas_path_npy> -out_dir <save data path>

#Train model

python trainer.py --models <D/U/R>  -mw <minimal loss weight>  -lw <label loss weight>  -dp <data path>  -mp <smpl mesh path>  -rd <model_dir> -split <train_test_splitfile>

#Generate meshes

python generator.py --models  <D/U/R> -checkpoint <ckpt> 
