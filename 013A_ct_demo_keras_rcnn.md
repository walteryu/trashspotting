# Caltrans Innovation Station - Idea Entry

## Litter Detection with Automated Image Analysis

### Problem Statement

Public litter has large environmental, sustainability and livability impacts in California. It requires substantial resources and exposes workers to pickup litter on state highways. As a result, this project proposes to address these challenges by identifying and monitoring high trash areas more efficiently with automated image analysis.

### Idea Proposal

The proposal development of a custom image analysis model to automate litter detection from field data collection. This notebook is a demonstration of such a model using a general implementation of the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) image recognition algorithm.

However, additional development is necessary to refine the general implementation into one which specifically detects litter on roadways; for example, Keep America Beautiful has open-sourced their [custom model](https://github.com/isaychris/litter-detection-tensorflow) for detecting litter on surface streets using Google Street View imagery.

### Benefits

Automated litter detection to identify more efficiently will have the following benefits:
1. Save Resources - Automate identification and monitoring of high trash areas and deploy cleanup crews more efficiently.
2. Minimize Worker Exposure - Cleanup crews will be deployed to high trash areas and minimize routine maintenance and inspection while exposed to live traffic.
3. Continuous Monitoring - Automation will allow for continuous monitoring of high trash areas and identification of new ones which may surface over time.
4. Asset Management - Image analysis can be recorded with GIS coordinates and integrated into statewide asset management and inventory efforts.
5. Streamlined Reporting - Image analysis produces date/time, location and trash level at time of inspection which will streamline reporting efforts.
6. Photo/Video Analysis - Custom model can analyze photos/videos and could enable inspection by a variety of devices and sources. Example of video analysis is shown in this [tutorial](https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e).
7. Easy Field Data Collection - The model can analyze images/videos of varying quality; as a result, data can be collected using a variety of devices.
8. General Inspection - The custom model can be tailored to identify other objects which may reduce inspection time and worker exposure.
9. Aerial Imagery - Image analysis can be used to analyze aerial imagery which would be used to identify very high trash areas where litter may be visible. This [presentation](https://gisdaystanford2017.sched.com/event/CPtj/geospatial-machine-learning-in-the-cloud-with-earth-engine-tensorflow-and-google-cloud-platform) is an example of aerial imagery analysis.
10. Continuous Improvement - Model may refined and improved over time; in addition, other image analysis algorithms may be used besides Mask R-CNN which may have higher performance.
11. Reduce Litter - Ultimately, image analysis should streamline litter cleanup.

### Requirements

The following tasks will be necessary to develop a custom image analysis model:
1. Field Data Collection - A large set of representative photos will be needed to train the image analysis model; as a result, field photos will need to be collected from high trash areas.
2. Image Annotation -
2. Development - Once photos are collected, then they will be used to train the image analysis model and test with inspection photos; initial results will be used to refine and improve the model.
3. Deployment - Once the model has achieved good statistical accuracy (90-95%), then it will be deployed into production to analyze inspection photos.
4. Model Improvement - The model will need to be maintained and improved over time. Image analysis and data science are evolving fields; as a result, time is required on the upkeep of the deployed model and related technologies.

### Deliverables

Once developed, the custom model would consist of the following components:
1. Training Images - A large set of representative photos will be collected for model training.
2. Custom Model - Trained model can test inspection photos and incorporate other algorithms or technologies as the field evolves.
3. Related Technologies - Image analysis results may be developed into GIS layers and datasets. Model can be deployed to other sources such as mobile devices or internal websites using [TensorFlow.js](https://js.tensorflow.org/).

### Current Adoption
(https://github.com/isaychris/litter-detection-tensorflow)
(http://www.dot.ca.gov/dist4/rightofway/datacenter/survid.htm)
(http://dot.ca.gov/hq/maint/Pavement/Offices/Pavement_Management/PDF/2015-2016_SHS_Pavement_Condition_Summary_Final_12-31-18a.pdf)

### Workflow

Automated image analysis would consist of the following workflow steps:
1. Field Photo Collection - Photos will need to be collected from high trash areas and inspection areas to create two datasets; one to train the model and the other to test images for results.
2. Train Model and Test Images - Once training and test photos are collected, then they will be used train the model and test the inspection photos.
3. Analyze Results - Model training and testing will provide accuracy results which will be analyzed to evaluate model performance.
4. Deploy Crews - Identify high trash areas and deploy cleanup crews.
5. Continuous Monitoring - Once crews are deployed, then inspections can continue with additional photos for imagery analysis; result will be iterative feedback loop.

### Demo

The demonstration contains code to train a general image recognition model and test it with publicly available photos from the Don't Trash California [website] and Google Street View.

Please keep in mind...

### Citations

The demonstration below was created using the following technologies:
[Google TensorFlow](https://www.tensorflow.org/) - Image Analysis Library
[Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) - Hosting Environment for this Notebook
[TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) - Object Detection Library.
