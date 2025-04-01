mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

conda create -n rpd python=3.10 -y
conda rpd openvla

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
git clone https://github.com/openvla/openvla.git
git clone https://github.com/krohling/refined-policy-distillation.git

cd openvla
pip install -e .

cd ../LIBERO
pip install -e .

cd ../openvla
pip install -r experiments/robot/libero/libero_requirements.txt

cd ../refined-policy-distillation
pip install -r requirements-ppo.txt
