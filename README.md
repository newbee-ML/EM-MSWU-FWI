# Multi-stage Warm-up Seismic Full Waveform Inversion Informed by Electromagnetic Data

Author: Hongtao Wang [1,3], and Dikun Yang [1,2]

Affiliation:

[1] Department of Earth and Space Sciences, Southern University of Science and Technology, Shenzhen, China.

[2] Guangdong Provincial Key Laboratory of Geophysical High-resolution Imaging Technology, Southern University of Science and Technology, Shenzhen, China.

[3] School of Mathematics and Statistics, Xi'an Jiaotong University, Xi'an, China.

Correspondingauthor: Dikun Yang (yangdikun@gmail.com)

Code Author: Hongtao Wang (colin315wht@gmail.com)

---

## 1 Methodology of EM-MSWU FWI
Full waveform inversion (FWI) provides high-resolution velocity models by exploiting the complete information embedded in the received seismic waveforms. However, FWI's advantage comes at the risk of local minimum and non-convergence when there is insufficient information about the large-scale or low-frequency structure to build a good initial model. EM's diffusive nature makes it an independent and low-cost source of information to complement the low-frequency component of seismic data. We propose a new FWI framework that enables the utilization of the EM's resistivity inversion result to prepare the FWI and forces the FWI to find the large-scale background before fine-tuning the details. Our implementation includes three stages. The first stage inverts EM data to obtain a blurred resistivity image of the subsurface. Then, the pixels on the resistivity image are clustered into dozens of geological domains that are likely to have a constant velocity value. Next, a preliminary FWI is carried out to roughly fit the seismic data by searching velocity values for these geological domains. Finally, the resultant velocity model bearing the EM's information and the low-frequency seismic information is refined by a full-bandwidth FWI that pursues the highest possible resolution. Tested on the Marmousi model, our approach over-performs some conventional FWI methods that solely rely on seismic data and behaves similarly to the theoretical smoothed initial model because an EM survey can be physically considered as a moving-window averaging of the subsurface structure. Our finding highlights the importance of multi-physical integration in subsurface imaging and showcases a straightforward workflow for the joint acquisition and analysis of seismic and EM data in practice.

## 2 Run the code

### 2.1 Check your source data 
> Before run the code, you should prepare both the true velocity model of Marmousi and the corresponding resistivity model of CSEM inversion.

Please check the path of your workspace in the file `config.py` 

> Also you need prepare the python packages
> 
Please run the cmd in your conda `pip install -r requirements.txt` 

### 2.2 Run Cascade FWI 
```Python
python models/emmswu_inversion/emmswu_FWI.py --save_group CascadeFWI  --vel_model_name mar_small --rerun_forward 1 --device 0 --noise_SNR 10000 --wu_cluster_k 100 --wu_train_lr 20 --wu_max_epoch 20 --wu_loss L1 --wu_train_bs 8 --wu_data_norm 0 --c_train_lr 30 --c_loss L1 --c_max_epoch 1000 --c_train_bs 8 --c_data_norm 0 --c_opt_stg MSLR --seed 8
```

### 2.3 Run FWIGAN 
```Python
python models/FWIGAN/FWIGAN_main.py --save_group FWIGAN --vel_model_name mar_small --rerun_forward 1 --device 0 --noise_SNR 10000 --init_method line --max_epoch 1000 --train_lr 30 --loss W1 --data_norm 0 
```

### 2.4 Run Classical FWI 
```Python
# L2 loss
python models/FWIGAN/FWIGAN_main.py --save_group FWI_L2 --vel_model_name mar_small --rerun_forward 1 --device 0 --noise_SNR 10000 --init_method line --max_epoch 1000 --train_lr 30 --loss W1 --data_norm 0 
# L1 loss
python models/classic_FWI/inversion_FWI.py --save_group FWI_L1 --vel_model_name mar_small --rerun_forward 1 --device 0 --init_method line --max_epoch 1000 --train_lr 20 --loss L1 --train_bs 8 --data_norm 0
# W1 Loss
python models/classic_FWI/inversion_FWI.py --save_group FWI_W1 --vel_model_name mar_small --rerun_forward 1 --device 0 --init_method line --max_epoch 1000 --train_lr 20 --loss W1 --train_bs 8 --data_norm 0
```

### 2.5 Run Classical FWI with SA Initial Model
```Python
python models/classic_FWI/inversion_FWI.py --save_group FWI_SA_init --vel_model_name mar_small --rerun_forward 1 --device 0 --init_method SA --max_epoch 1000 --train_lr 20 --loss L1 --train_bs 8 --data_norm 0
```

## 3 Results of all models

> Save in the folder `source_models` 

1. The inversion models of comparison methods: `source_models\inversion_models`

2. The inversion models of various initial models: `source_models\inversion_models`