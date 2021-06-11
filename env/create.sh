#conda create -y -n dst python=3.8 && \
#conda install -y -n dst pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia && \
#conda install -y -n dst Cython && \
conda run -n dst --no-capture-output python -m pip install git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]
