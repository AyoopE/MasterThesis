This repository was created for:
Master Thesis: Enhanced analysis of circulating tumor DNA in pancreatic cancer using machine learning
Student: Ayoop Ali Deeb Elkafarna


The repository contains the following 5 files other than README.md:

1. First_Classification_Problem.py
This is a python file for all experiments for the first classification problem.

2. Second_Classification_Problem.py
This is a python file for all experiments for the second classification problem.

3. Regression_Problem.py
This is a python file for all experiments for the regression problem.

4. all_functions.py
This is a python file contains all the functions needed for python scripts for the three machine learning problems mentioned above.

5. requirements.txt
This is a text file that contains the work environment packages in python where I performed the experiments.
(This file contains the output of the command: pip freeze > requirements.txt)


######## The datasets ########
The datasets used in experiments exist on unix server in this folder: /home/prosjekt/BMDLab/data/pactact/Master_Thesis_Ayoop

There are six csv files inside the folder:

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
The dataset in the second file (Second_Classification_Problem_data_part2.csv) contains samples  to be predicted by the created model.

The following files are for the regression problem:
Regression_Problem_data_part1.csv
Regression_Problem_data_part2.csv
Where the regression model built on the dataset in the first file (Regression_Problem_data_part1.csv)
The dataset in the second file (Regression_Problem_data_part2.csv) contains samples  to be predicted by the created model.


##### Running python scripts ########

Five Python libraries are needed for python scripts in this repository:
NumPy v1.23.5
Matplotlib v3.7.1
Scikit-learn v1.2.2
Pandas v1.5.3
SciPy v1.10.1

Python scripts can be run directly. But there are also comments almost for each command for more clarification.


To run pyhton script for the first classification problem, the following files are needed to be in the same working directory:
First_Classification_Problem.py
First_Classification_Problem_data_part1.csv
First_Classification_Problem_data_part2.csv
all_functions.py
Create an empty folder with the name "results" 
Create an empty folder with the name "images"

To run pyhton script for the second classification problem, the following files are needed to be in the same working directory:
Second_Classification_Problem.py
Second_Classification_Problem_data_part1.csv
Second_Classification_Problem_data_part2.csv
all_functions.py
Create an empty folder with the name "results" 
Create an empty folder with the name "images"

To run pyhton script for the regression problem, the following files are needed to be in the same working directory:
Regression_Problem.py
Regression_Problem_data_part1.csv
Regression_Problem_data_part2.csv
all_functions.py
Create an empty folder with the name "results" 
Create an empty folder with the name "images"


OR Breifly:
You can have all python files and datasets in the same working directory and then you just have to create two empty folders for all machine learning problems with the names "results" and "images"
