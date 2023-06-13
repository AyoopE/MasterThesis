This repository is created for:
Master Thesis: Enhanced analysis of circulating tumor DNA in pancreatic cancer using machine learning
Student: Ayoop Ali Deeb Elkafarna


The repository contains the follwoing 4 files, other than README.md:

1. First_Classification_Problem.py
This is a python file for all experiments for first classification problem.

2. Second_Classification_Problem.py
This is a python file for all experiments for second classification problem.

3. Regression_Problem.py
This is a python file for all experiments for regression problem.

4. all_functions.py
This is a python file for all functions needed for python scripts for the three machine learning problems mentioned over.



The datasets exist in the folder: UnixServer\\prosjekt\BMDLab\data\pactact\Master_Thesis_Ayoop
Inside the folder are 6 csv files:

First_Classification_Problem_data_part1.csv
First_Classification_Problem_data_part2.csv
Second_Classification_Problem_data_part1.csv
Second_Classification_Problem_data_part2.csv
Regression_Problem_data_part1.csv
Regression_Problem_data_part2.csv

The following files are for the first classification problem:
First_Classification_Problem_data_part1.csv
First_Classification_Problem_data_part2.csv
Where the first classification model was built on the dataset in the first file (First_Classification_Problem_data_part1.csv)
The dataset in the second file (First_Classification_Problem_data_part2.csv) contains samples to be predicted by the created model.

The following files are for the second classification problem:
Second_Classification_Problem_data_part1.csv
Second_Classification_Problem_data_part2.csv
Where the second classification model was built on the dataset in the first file (Second_Classification_Problem_data_part1.csv)
The dataset in the second file (Second_Classification_Problem_data_part2.csv) contains samples  to be predicted by the created model.

The following files are for the regression problem:
Regression_Problem_data_part1.csv
Regression_Problem_data_part2.csv
Where the regression model built on the dataset in the first file (Regression_Problem_data_part1.csv)
The dataset in the second file (Regression_Problem_data_part2.csv) contains samples  to be predicted by the created model.



To run pyhton script for the first classification problem, the following files are needed to be in the same working directory:
First_Classification_Problem.py
First_Classification_Problem_data_part1.csv
First_Classification_Problem_data_part2.csv
all_functions.py
Create empty folder with name "results" 
Create empty folder with name "images"

To run pyhton script for the second classification problem, the following files are needed to be in same the working directory:
Second_Classification_Problem.py
Second_Classification_Problem_data_part1.csv
Second_Classification_Problem_data_part2.csv
all_functions.py
Create empty folder with name "results" 
Create empty folder with name "images"

To run pyhton script for the regression problem, the following files are needed to be in the same working directory:
Regression_Problem.py
Regression_Problem_data_part1.csv
Regression_Problem_data_part2.csv
all_functions.py
Create empty folder with name "results" 
Create empty folder with name "images"


OR Breifly:
you can have all python files and datasets in the same working directory and then you have just to create two empty folders for all ML problems with names "results" and "images"
