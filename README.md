# Time-Series-Imputation-Analysis

### requirements:

注意torch版本应与cuda版本对应。可去torch官网下载对应版本。例如：

```aiignore
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

下载完对应的torch之后，其余的库也应当与torch版本对应。例如：

```aiignore
darts==0.38.0
einops==0.8.0
lightgbm==4.6.0
local-attention==1.9.14
matplotlib==3.10.7
MissForest==4.2.3
numpy==2.2.6
pmdarima==2.0.4
PyWavelets==1.8.0
pandas==2.3.3
patool==1.12
reformer-pytorch==1.4.4
statsmodels==0.14.5
scikit-learn==1.7.2
scipy==1.15.3
sktime==0.39.0
sympy==1.14.0
toml==0.10.2
tqdm==4.64.1
xgboost==2.0.3
timesfm==2.0.0
django-layers-hr
```

### 基础模型运行部分：
所有的基础模型均为huggingface上对应版本的checkpoint。目前的逻辑为先将模型文件本地保存，再调用本地模型运行。

#### chronos:

运行chronos_2需要：

```aiignore
pip install "chronos-forecasting>=2.0"
`pip install -U sagemaker`
```

#### karios:

运行Karios_23m需要：

`pip install tsfm`

#### moirai:

运行moirai_2p0_r_small需要引入uni2ts库，目前没法pip获取，可从github上clone下来后安装：

```aiignore
git clone https://github.com/SalesforceAIResearch/uni2ts.git
cd uni2ts
pip install -e '.[notebook]'
```

#### sundial:

运行sundial_base_128m需要：

`pip install transformers==4.40.1`


#### TimesFM:

运行timesfm_2p0_500m需要1.3版本的timesfm库,默认pip得到的就是：

`pip install timesfm`

运行timesfm_2p5_200m需要2.0版本的timesfm库，，目前没法pip获取，可从github上clone下来后安装：

```aiignore
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .
```
注意：timesfm2.0版本无法向下兼容，因此运行timesfm2.5版本模型和老版本模型需要不同的虚拟环境


#### VisionTS

运行VisionTSpp需要：
`pip install visionts`

