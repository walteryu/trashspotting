# Trashspotting - Image Recognition Framework Evaluation

## Collection of Tutorials to Evaluate Image Recognition Models

### Purpose

Evaluate image recognition models to develop custom model for litter detection in California. Public litter has large environmental, sustainability and livability impacts in California. As a result, this project seeks to help address the issue with deep learning.

### Implementation (In Progress)

Project implementation uses the technologies listed below and includes the necessary files and datasets to deploy the model:

1. [Google TensorFlow](https://www.tensorflow.org/)
2. [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
3. [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
4. [Docker - TensorFlow Image](https://hub.docker.com/r/tensorflow/tensorflow/)
5. [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html)

Each module is intended to run in this [Colaboratory notebook](https://colab.research.google.com/drive/1gy4IcA6Kmasez6TXu1NM6fR3YWdKKr1f), so clone the notebook to your Google Drive account and paste desired module to run it.

Each major tutorial is organized into its own notebook module and cited accordingly; also included are annotated Google Street View images with CA highway litter for model training/testing.

### Best Practices

TensorFlow requires a good amount of setup and configuration, so the tools listed below are recommended to make it more approachable and easier to build models:

1. [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) - Free Jupyter Notebook environment from Google specialized for TensorFlow; specifically, Google provides free GPU-enabled notebooks!
2. [Docker](https://hub.docker.com/r/tensorflow/tensorflow/) - TensorFlow loads up in Jupyter Notebook; saves time in local installation of all modules and dependencies.
3. [Keras](https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd) - Example tutorial demonstrating image recognition with Keras and TensorFlow.
4. [Jupyter Notebook](https://medium.com/@margaretmz/anaconda-jupyter-notebook-tensorflow-and-keras-b91f381405f8) - Example tutorial demonstrating notebook setup with Keras and TensorFlow.
5. [Anaconda](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46) - Example tutorial shows how to install Jupyter Notebook with the Anaconda package manager.

### Installation

Clone Github repository, then run locally or with Google Colaboratory. Local installation will require TensorFlow and all dependencies; as a result, Colaboratory or Docker image are highly recommended.

### Initial Results

Initial results are shared in this [Colaboratory Notebook](https://colab.research.google.com/drive/1E1HDwNLs1HAyCPifc1QDrM1x17lUEXb1); it is based on this [tutorial](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/) with Keras and the Mask R-CNN Algorithm to analyze publicly available photos and Google Street View imagery.

### Citations - Tutorials and Code Repositories (In Progress)

Module 2: [Custom Model in Colab - Medium Article](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)\
Module 2: [Custom Model in Colab - Github](https://hackernoon.com/object-detection-in-google-colab-with-custom-dataset-5a7bb2b0e97e)\
Module 3: [Toy Detector - Medium Article](https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95)\
Module 3: [Toy Detector - Medium Article](https://github.com/walteryu/Deep-Learning/tree/master/tensorflow_toy_detector)\
Module 4: [KAB Litter Detection Algorithm - Github](https://github.com/isaychris/litter-detection-tensorflow)\
Module 5: [KAB Litter Detection Algorithm - Colab Notebook](https://github.com/isaychris/litter-detection-tensorflow)\
Module 6: [Chess Object Detector - O'Reilly Media](https://www.oreilly.com/ideas/object-detection-with-tensorflow)\
Module 6: [Chess Object Detector - Github](https://github.com/wagonhelm/TF_ObjectDetection_API)\
Module 7: [Chess Object Detector - Docker Script](https://www.oreilly.com/ideas/object-detection-with-tensorflow)\
Module 8: [Raccoon Dataset - Medium Article](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)\
Module 8: [Raccoon Dataset - Github](https://github.com/datitran/raccoon_dataset)\
Module 9: [Millennium Falcon Detector - Medium Article](https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e)\
Module 9: [Millennium Falcon Detector - Github](https://github.com/bourdakos1/Custom-Object-Detection)\
Module 10: [Image AI - Medium Article](https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606)\
Module 10: [Image AI - Github](https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Prediction)\
Module 11: [RetinaNet with Keras - Github](https://github.com/fizyr/keras-retinanet)\
Module 12: [Mask R-CNN with Keras - Article](https://www.dlology.com/blog/how-to-run-object-detection-and-segmentation-on-video-fast-for-free/)\
Module 12: [Mask R-CNN with Keras - Github](https://github.com/Tony607/colab-mask-rcnn)
