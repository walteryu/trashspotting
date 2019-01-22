# Caltrans Innovation Station - Idea Entry

## Litter Detection with Automated Image Analysis

### Problem Statement

Public litter has large environmental, sustainability and livability impacts in California. It requires substantial resources and exposes workers to pickup litter on state highways. As a result, this project proposes to address these challenges by identifying and monitoring high trash areas more efficiently with automated image analysis.

### Idea Proposal

This notebook proposes to continue development of a custom image analysis model for automating litter detection and includes a demonstration of general model using the [Mask R-CNN](https://github.com/matterport/Mask_RCNN) image recognition algorithm.

The additional development would refine the general implementation into one which specifically detects litter on roadways; for example, Keep America Beautiful has commissioned and open-sourced their [custom model](https://github.com/isaychris/litter-detection-tensorflow) for detecting litter on surface streets using Google Street View imagery.

### Benefits

Automated litter detection will have the following benefits:
1. Save Resources - Automate identification of high trash areas and deploy cleanup crews to address them more safely and efficiently.
2. Minimize Worker Exposure - Cleanup crews will be deployed to high trash areas and minimize routine maintenance and inspection while exposed to live traffic.
3. Continuous Monitoring - Automation will provide continuous monitoring of high trash areas and identification of new ones which may surface over time.
4. Asset Management - Image analysis results can be incorporated into statewide GIS layers, asset management and inventory efforts.
5. Streamlined Reporting - Image analysis produces date/time, location and trash level at time of inspection which will streamline reporting efforts.
6. Photo/Video Analysis - Custom model can analyze photos/videos and could enable inspection by a variety of devices and sources. Example of video analysis is shown in this [tutorial](https://medium.freecodecamp.org/tracking-the-millenium-falcon-with-tensorflow-c8c86419225e).
7. Streamline Data Collection - The model can analyze images/videos of varying quality; as a result, data can be collected more easily using a variety of devices.
8. General Inspection - The custom model can be tailored to identify other objects which may reduce inspection time and worker exposure.
9. Aerial Imagery - Image analysis can be used to analyze aerial imagery which would be used to identify very high trash areas where litter may be visible. This [presentation](https://gisdaystanford2017.sched.com/event/CPtj/geospatial-machine-learning-in-the-cloud-with-earth-engine-tensorflow-and-google-cloud-platform) is an example of aerial imagery analysis.
10. Continuous Improvement - Model may refined and improved over time; in addition, other image analysis algorithms may be used besides Mask R-CNN which may have higher performance.
11. Reduce Litter - Ultimately, image analysis should help reduce litter.

### Requirements

The following tasks will be necessary to develop a custom image analysis model:
1. Field Data Collection - Representative photos will be needed to collected from high trash generating areas for training the custom model.
2. Image Annotation -Photos in the training dataset will need to be annotated with bounding boxes and labels for incorporating into the custom model.
3. Development - Once photos are collected and processed, then they will be used to train the image analysis model and test with inspection photos; initial results will be used to refine the model for deployment.
4. Deployment - Once the model has achieved good statistical accuracy (90-95%), then it will be deployed into production to analyze inspection photos.
5. Model Improvement - The model will need to be maintained and improved over time. Image analysis and data science are evolving fields; as a result, time is required on the upkeep of the deployed model and related technologies.

### Deliverables

Once developed, the custom model will consist of the following components:
1. Training Images - A large set of representative photos from high trash generating areas.
2. Custom Model - Trained model can test inspection photos and incorporate other algorithms or technologies as the field evolves.
3. Related Technologies - Image analysis results may be developed into GIS layers and datasets. Model can be deployed to other sources such as mobile devices or internal websites using [TensorFlow.js](https://js.tensorflow.org/).

### Current Adoption

Similar technologies are already implemented by academia and industry:
1. [Keep America Beautiful (KAB)](https://github.com/isaychris/litter-detection-tensorflow) - KAB commissioned development of a litter detection model and open sourced their results.
2. [SJSU Clean Streets Project](https://www.computer.org/csdl/proceedings/bigdataservice/2017/6318/00/07944927.pdf) - SJSU project which use analyzes field data collected from street sweepers to streamline routine street cleaning in collaboration with the City of San Jose.
3. [SFEI UAV Project](https://www.sfei.org/projects/uav#sthash.Z6k5RvII.dpbs) - SFEI is collecting aerial imagery with drones (UAV) and analyzing the imagery with image recognition.

The Department is starting to utilize LIDAR and image analysis as follows:
* [District 4 LIDAR Project](http://www.dot.ca.gov/dist4/rightofway/datacenter/survid.htm) - D4 is deploying LIDAR imagery collection within the SF Bay Area.
* [HQ Pavement Program](http://dot.ca.gov/hq/maint/Pavement/Offices/Pavement_Management/PDF/2015-2016_SHS_Pavement_Condition_Summary_Final_12-31-18a.pdf) - HQ is starting to incorporate image analysis into Pavement Condition inspection.

### Workflow

Automated image analysis will consist of the following workflow steps:
1. Field Photo Collection - Photos will need to be collected from high trash areas and inspection areas to create two datasets; one to train the model and the other to test images for results.
2. Train Model and Test Images - Once training and test photos are collected, then they will be used train the model and test the inspection photos.
3. Analyze Results - Model training and testing will provide accuracy results which will be analyzed to evaluate model performance.
4. Deploy Crews - Identify high trash areas and deploy cleanup crews.
5. Continuous Monitoring - Once crews are deployed, then inspections can continue with additional photos for imagery analysis; result will be iterative feedback loop.

### Demo

The demonstration contains code to train a general image recognition model and test it with publicly available photos from the Don't Trash California [website](http://www.dot.ca.gov/dist11/trash/pics.html), [Pixabay](https://pixabay.com/en/photos/trash/) and Google Street View.

Please keep in mind that image analysis is a difficult computation problem, and research is typically completed by a development team. This project is currently a solo effort; hence, this proposal aims to continue and enhance the development process.

As a result, initial results of this demo have the limitations listed below and erroneous results are included to illustrate the potential of this technology and need for additional development:
1. Missed Objects - This demo uses a generic model trained for everyday objects; as a result, it does not detect objects not specifically included as a training class.
2. Misclassification - The generic model is not tailed for litter detection, so it misclassifies objects, e.g. litter as people or animals.
3. Multiple Objects - The generic model is trained on a significant dataset; as a result, the custom model will need to be trained from scratch.

### Citations

The demonstration below was created using the following technologies:
1. [Google TensorFlow](https://www.tensorflow.org/) - Image Analysis Library
2. [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) - Hosting Environment for this Notebook
3. [TensorFlow Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection) - Object Detection Library
