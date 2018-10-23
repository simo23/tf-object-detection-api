## How to setup tf-object-detection-api, NVIDIA drivers, CUDA, cuDNN, Anaconda and Tensorflow 

**Note 1**: this has been tested on Ubuntu 16.04, python 2.7, NVIDIA GPUs, tensorflow 1.9, CUDA 9.0, cuDNN 7.0 but should work for other versions with small changes 

**Note 2**: you can install everything with sudo rights or without them ( we will install everything locally ). If you use sudo be very careful! 

**Note 3**: if you have already CUDA installed and working you can skip to step 4

#### 1. Download and install CUDA 9.0 and/or NVIDIA drivers

The installation process is almost automatic. You have just to launch the execution of one file. Before doing that, you should decide what to do!

##### What to install?
The file containing the CUDA libraries actually contains also the NVIDIA drivers. You can decide to:
* **If you can use sudo**: you can also install/update your NVIDIA drivers. Be aware that if you or anyone else has another installation of the NVIDIA drivers or CUDA you should be very careful. If you decide to install the drivers the older ones will be removed. Select YES to install them only if you are confident about this
* **If you cannot use sudo**: you can just install the CUDA libraries. Select NO when the process will ask to install the drivers

##### Where to install CUDA?
The default installation path is "/usr/local/cuda".
* **If you can use sudo**: if you want to install them for everyone who is using this computer keep the default path "/usr/local/cuda". Be aware that if you or anyone else has another installation of CUDA in this path it will be overwritten
* **If you cannot use sudo or you want to install them somewhere else**: create a folder where you want. Then you will need to insert it in the terminal later. Example: create a folder in /home/nvidia_libraries/cuda-9.0/ and then write "/home/nvidia_libraries/cuda-9.0/" when asked where to install CUDA.

##### CUDA and/or NVIDIA drivers installation

Then the process is very straightforward. We will download the file and launch it. It will ask what and where to install some things. You should be careful to install only the things that you really need! The process will ask to install:

1. NVIDIA drivers ( you can say NO )
2. CUDA referred as Toolkit or CUDA Toolkit ( you should say YES )
3. CUDA samples ( you can say NO )

Go ahead and download the file (here the link is for CUDA 9.0) with:

     cd
     cd Downloads
     wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run


* **If you can use sudo**: just run the installation with sudo rights

      sudo sh cuda_9.0.176_384.81_linux.run

* **If you cannot use sudo or you want to install them somewhere else**: create the folder where to install CUDA and launch the executable

      cd
      cd ~/
      mkdir nvidia_libraries
      mkdir nvidia_libraries/cuda-9.0

      chmod +x cuda_9.0.176_384.81_linux.run
      ./cuda_9.0.176_384.81_linux.run

#### 2. Download and install cuDNN 7.0 ( required to run CUDA 9.0 )

Installing cuDNN is just a matter of copying some files in the CUDA directories. You have to create a profile on the NVIDIA website and then navigate to the download page of cuDNN (https://developer.nvidia.com/rdp/cudnn-archive). Download cuDNN 7.0 for linux by clicking on the link. Put the file in ~/Downloads and extract it

     cd
     cd Downloads
     tar -xzvf cudnn-9.0-linux-x64-v7.tgz

##### cuDNN installation

Installing cuDNN is just a matter of copying some files in the CUDA directories. You must download the files and then find there CUDA was installed.

* If you have installed CUDA in **/usr/local/cuda**:

      sudo cp cuda/include/cudnn.h /usr/local/cuda/include
      sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
      sudo chmod a+r /usr/local/cuda/include/cudnn.h

* If you have installed CUDA **somewhere else** ( in /home/nvidia_libraries/cuda-9.0/ for example ):

      cp cuda/include/cudnn.h /home/nvidia_libraries/cuda-9.0/include
      cp cuda/lib64/libcudnn* /home/nvidia_libraries/cuda-9.0/lib64
      chmod a+r /home/nvidia_libraries/cuda-9.0/include/cudnn.h

#### 3. Set enviroment variables for CUDA libraries

Once you have installed CUDA and cuDNN you have to say to your system where to find them. We will modify the environmental variables called LD_LIBRARY_PATH and PATH to include the locations of CUDA:

     nano ~/.bashrc

Copy and paste the following lines at the bottom (assuming that CUDA is in "/home/nvidia_libraries/cuda-9.0/")

     export LD_LIBRARY_PATH="/home/nvidia_libraries/cuda-9.0/lib64:$LD_LIBRARY_PATH"
     export PATH="/home/nvidia_libraries/cuda-9.0/bin:$PATH"

Close the file and the terminal in order to make it effective.

#### 4. Download and install Anaconda

Anaconda is a very powerful tool which allows you to create organized environments where to install your things without issues. We need to download a single file and run it. Check all the things that will be asked during the process but everything should be kept as default.

     cd 
     cd Downloads
     wget https://repo.anaconda.com/archive/Anaconda2-5.2.0-Linux-x86_64.sh
     bash ~/Downloads/Anaconda2-5.2.0-Linux-x86_64.sh

Close and re-open the terminal to make it effective. 

##### Create conda environment

You can create an environment called "tf" by firing:

     conda create --name tf python=2.7 
     source activate tf

#### 5. Install Tensorflow and other packages

This is a one-liner. Here we will install the 1.9 because it is the latest available. You have just to decide if you want to install tensorflow to use it with GPUs or not:

* If you want to use **GPUs**:

      pip install tensorflow-gpu==1.9 

* If you want to use a **CPU**:

      pip install tensorflow==1.9 

If there is an error with pip do python -m pip install --upgrade pip and retry. Now install some other useful packages: 

     pip install matplotlib opencv-python 

#### 6. Download and setup the tf-object-detection-api

Clone the repository where you want. In ~/tf-object-detection-api for example:

     cd 
     cd ~/
     git clone https://github.com/simo23/tf-object-detection-api

Install the object_detection package:

     cd
     cd ~/tf-object-detection-api/object_detection
     python setup.py install

Install cocoapi:

     cd
     cd ~/tf-object-detection-api/object_detection/cocoapi/PythonAPI
     make
     python setup.py install

Set the environment variable PYTHONPATH to include object_detection and slim folders:

     nano ~/.bashrc

Copy and paste this line at the bottom:

     export PYTHONPATH=$PYTHONPATH:~/tf-object-detection-api:~/tf-object-detection-api/object_detection/slim

#### 7. Test the installation

If everything is installed properly you should be able to perform detection on a sample image straight away. Try to export the graph of the pre-trained model as:

     python export_inference_graph.py --conf=./experiments/exp_1/model.config --ckpt=./object_detection/weights/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt --output_dir=./experiments/exp_1/exported_graph/ --iou_th=0.1

Or if you want to use a GPU, say the number 0:

     CUDA_VISIBLE_DEVICES=0 python export_inference_graph.py --conf=./experiments/exp_1/model.config --ckpt=./object_detection/weights/ssdlite_mobilenet_v2_coco_2018_05_09/model.ckpt --output_dir=./experiments/exp_1/exported_graph/ --iou_th=0.1

Now run the detection with: 

     python detect_jpg_folder.py -i=./data/sample_jpg_folder/ -f=./experiments/exp_1/exported_graph/frozen_inference_graph.pb -d=0

If an image with a detected "person" and "surfboard" pops up you are good to go. Press "q" to close it. 

#### Done! 

Close every terminal in order to make everything effective
