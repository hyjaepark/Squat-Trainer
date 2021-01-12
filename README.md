# Squat-Project
Personal trainer for workout pose adjustments

![text](demo.gif)


# How to make it work

## 1. Scrape image 
I took around 500 photos of our team doing the squat exercise. Included Google images as well.

## 2. Use Labeling program to identify body parts 
Used https://github.com/tzutalin/labelImg to label 5 body parts.

## 3. Save labels to XML

> python xml_to_csv.py

## 4. Create TF records

> python generate_tfrecord.py

## 5. Training

> python train.py

## 6. Import train result, detect movement and show output
Requires webcam to use

> python squat_trainer_version_0128.py

