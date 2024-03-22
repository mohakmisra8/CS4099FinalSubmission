# Use the TensorFlow image as a base
FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

# Set environment variables
ENV DEBIAN_FRONTEND="noninteractive"

# Install locales and time zone data
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt-get install --yes --no-install-recommends locales tzdata && \
    echo "en_GB.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    rm -rf /var/lib/apt/lists/*

# Add the PATH for user-installed binaries
ENV PATH="/root/.local/bin:${PATH}"

# Copy the requirements file and install dependencies
# COPY requirements.txt ./
RUN pip3 install --user --upgrade --disable-pip-version-check pip
RUN pip3 install --user --no-cache-dir --disable-pip-version-check --root-user-action=ignore Flask requests
RUN pip install albumentations==0.4.6

# Install additional Python packages
RUN pip3 install --user --no-cache-dir --disable-pip-version-check \
    SimpleITK tqdm matplotlib nibabel albumentations pydicom nibabel torch torchvision opencv-python scikit-image nilearn && \
    pip install git+https://github.com/shijianjian/EfficientNet-PyTorch-3D && \
    pip install einops && \
    pip install segmentation-models-3D

# Create a directory inside the container to store the dataset
RUN mkdir -p /dataset

# Copy the dataset from your host machine to the container

# Install bc to handle symbolic links
RUN apt-get update && apt-get install -y bc
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the split_data.sh script
# COPY split_data.sh /usr/local/bin/split_data.sh
# RUN chmod +x /usr/local/bin/split_data.sh

# Start your application (Jupyter Lab, in this case)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--port=8888", "--NotebookApp.token=''", "--NotebookApp.password=''"]
