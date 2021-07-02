conda create -y -n dst python=3.8 && \
conda install -y -n dst pytorch=1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia && \
conda install -y -n dst Cython=0.29.23 && \
conda run -n dst --no-capture-output pip install -r requirements.txt
