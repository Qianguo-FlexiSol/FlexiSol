1.INSTALL

conda env create -f env.yml
  conda activate FlexSol
  python -c "
from transformers import AutoTokenizer, EsmModel
m='facebook/esm2_t33_650M_UR50D'
AutoTokenizer.from_pretrained(m)
EsmModel.from_pretrained(m)
"
env.yml::
name: solubility
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.1.2
  - pytorch-cuda=12.1
  - pip
  - pip:
    - transformers==4.40.0
    - datasets==2.18.0
    - scikit-learn==1.4.0
    - pandas==2.2.0
    - numpy==1.26.3
    - matplotlib==3.8.2
    - seaborn==0.13.2
    - tqdm==4.66.1
    - xgboost==2.0.3
    - lightgbm==4.3.0
    - jupyterlab==4.1.0

	  
  2.USAGE

python predict_single.py 
