#!/bin/bash
source "/home/zhexi/anaconda3/etc/profile.d/conda.sh"
eval "$(mamba shell hook --shell bash)"
export DATA_DIR="./mydata"
export TRANSFORMERS_CACHE="/home/zhexi/project/easyhoi/.cache"
:<<"END"
echo -e "\e[1;33;5m========== step1: HAMER START ==========\e[0m"
conda activate hamer
echo -e "\e[1;35;5m 手部重建\e[0m"
python preprocess/recon_hand.py --data_dir $DATA_DIR

echo -e "\e[1;33;5m========== step2: LISA START ==========\e[0m"
mamba activate lisa
echo -e "\e[1;35;5m 手部检测分割\e[0m"
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py --seg_hand --skip --load_in_8bit --data_dir $DATA_DIR
echo -e "\e[1;36;5m 物体检测分割\e[0m"
CUDA_VISIBLE_DEVICES=0 python preprocess/lisa_ho_detect.py --skip --load_in_8bit --data_dir $DATA_DIR

echo -e "\e[1;33;5m========== step3: AFFORDANCE DIFFUSION START ==========\e[0m"
echo -e "\e[1;35;5m 补全物体\e[0m"
mamba activate afford_diff
python preprocess/inpaint.py --data_dir $DATA_DIR --save_dir $DATA_DIR/obj_recon/ --img_folder images --inpaint --skip

echo -e "\e[1;33;5m========== Step 4: Segment inpainted obj get the inpainted mask ==========\e[0m"
echo -e "\e[1;35;5m 获取inpaint mask\e[0m"
conda activate easyhoi
python preprocess/seg_image.py --data_dir $DATA_DIR

echo -e "\e[1;33;5m========== Step 5: Reconstruct obj  ==========\e[0m"
echo -e "\e[1;35;5 调用Tripo3D api重建物体\e[0m"
conda activate base
python preprocess/tripo3d_gen.py --data_dir $DATA_DIR
END
echo -e "\e[1;33;5m========== step6: 获取水密网络  ==========\e[0m"
echo -e "\e[1;35;5m FIXME:暂时跳过步骤 \e[0m"

echo -e "\e[1;33;5m========== stage2: Optimization!  ==========\e[0m"
echo -e "\e[1;35;5m 三阶段优化 \e[0m"
conda activate easyhoi
python src/optim_easyhoi.py -cn optim_teaser_tripo