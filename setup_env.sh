conda create --name sdfusion python=3.9 -y && conda activate sdfusion

# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# other stuffs
pip install h5py joblib trimesh scipy

# extract sdf
# sudo apt-get install libglu1-mesa

