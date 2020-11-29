import csv
import os
import shutil

file_names = set()

def get_names(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        line = 0
        for row in reader:
            if line != 0:
                name = row[0]
                if name in file_names:
                file_names.add("mips/"+name)
            line+=1

def get_files():

    for filename in file_names:
        shutil.copytree(filename, "data/"+filename)

get_names("/media/user1/cec674ba-5ca8-4568-b5bb-be95f6a2947d/cnivera_pharinsu_gwarren2/dl-mechanical-thrombectomy/data_titles_and_features.csv")
get_files()