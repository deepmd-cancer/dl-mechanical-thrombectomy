import numpy as np
import os
import csv
import re
import matplotlib.pyplot as plt
import tensorflow as tf

TICI_MAP = {"0":0,"1":1,"2": 2,"2a":3,"2b":4,"2c":5,"3":6}
sentence_len = 25

def get_mips_data():

    data_dict = dict()

    dir = "../data/mips"

    # storing patient images from dict (data id -> image)
    with os.scandir("../data/mips") as folders:
        for folder in folders:
            dirpath = dir + "/" + folder.name
            with os.scandir(dirpath) as numpydata:
                for file in numpydata:
                    filepath = dirpath + "/" + file.name
                    loaded = np.load(filepath)
                    data_dict[folder.name] = loaded
    return data_dict

def append_csv_features(image_dict):
    data_with_features = dict()
    csvpath = "./data_titles_and_features.csv"
    test_fraction = 0.2
    with open(csvpath) as file:
        output = []
        labels = []
        reader = csv.reader(file)
        line = 0
        word2num = dict()
        word2num["*UNK*"] = 0
        curr_idx = 1
        for row in reader:
            if line != 0:
                ## TODO might need to normalize pixels between 0 -> 1
                data = image_dict[row[0]] #string id of patient to image data

                data = np.divide(data, np.float32(3071))
                # (64, 256, 256) => (64, 256, 256, 1)
                data = tf.expand_dims(data, axis=3)
                flipped_data = data.numpy()
                for i in range(len(flipped_data)):
                    flipped_data[i] = tf.image.random_flip_left_right(flipped_data[i])

                occlusion = parse_occlusion(row[1]) # either "occlusion" or "not" (binary)
                old_status =  parse_old_status(row[2]) #either "", "acute", or "chronic" (one hot)
                vessels = row[6] # a sentence string
                location = row[7] # string (append to vessels)
                vessel_and_locations = parse_vessels_locations(vessels, location)

                # replacing words in vessel_and_locations with indices of vocab
                for i, word in enumerate(vessel_and_locations):
                    if word not in word2num:
                        word2num[word] = curr_idx
                        curr_idx += 1

                    vessel_and_locations[i] = word2num[word]

                passes = parse_passes(row[8]) # avg case is number, edge case is Rightside/Leftside, edge case2 is inconsistent text
                tici = parse_tici(row[9]) #either "0","1","2a","2b","2c","3", edge case is R/L
                gender = 1 if row[10].lower() == "female" else 0 # either "Male" or "Female"
                age = int(row[11]) # positive int
                if passes is not None and data.shape == (64, 256, 256, 1):
                    for p in passes:
                        for t in tici:
                            entry = [data,vessel_and_locations]
                            flipped_entry = [flipped_data, vessel_and_locations]
                            entry.append(occlusion)
                            entry.extend(old_status)
                            entry.append(gender)
                            entry.append(age)

                            flipped_entry.append(occlusion)
                            flipped_entry.extend(old_status)
                            flipped_entry.append(gender)
                            flipped_entry.append(age)

                            output.append(entry)
                            output.append(flipped_entry)
                            labels.append([p,t])
                            labels.append([p,t])
            line += 1

    output = np.array(output)
    labels = np.array(labels)
    indices = np.arange(output.shape[0])
    np.random.shuffle(indices)
    output = output[indices]
    labels = labels[indices]

    train_data = np.array(output[:int(len(output) * (1 - test_fraction))])
    train_labels = np.array(labels[:int(len(output) * (1 - test_fraction))])
    test_data = np.array(output[int(len(output) * (1 - test_fraction)):])
    test_labels = np.array(labels[int(len(output) * (1 - test_fraction)):])

    return train_data, train_labels, test_data, test_labels, len(word2num)


def parse_occlusion(is_occlusion):
    """
    A function to convert a string to a binary value
    :param is_occlusion: either "occlusion" or "not"
    :return: 1 if "occlusion", 0 if "not"
    """
    return 1 if is_occlusion.lower() == "occlusion" else 0


def parse_old_status(status):
    """
    Function that maps string, status, to a one hot vector representing the status.
    :param status: a string that is None/"", "acute", or "chronic"
    :return: [1,0,0], [0,1,0], [0,0,1] in each respective case.
    """
    status = status.lower()
    if not status: # none
        return [1,0,0]
    elif status == "acute": # acute
        return [0,1,0]
    else: # chronic
        return [0,0,1]

def parse_vessels_locations(vessel, location):
    """
    Parses vessel, removing any alphanumeric characters, and concatenates with location.
    :param vessel: a sentence string
    :param location: a location string
    :return: the parsed and concatenated string
    """

    if not vessel:
        vessel = ""
    if not location:
        location = ""
    #check none
    vessel = vessel.lower()
    vessel = re.sub('[^0-9a-zA-Z]+', ' ', vessel)
    location = location.lower()
    concatenated = vessel + " " + location
    output = concatenated.split()
    while len(output) < sentence_len:
        output.append("*UNK*")
    # add elements to split until length is == sentence_length

    return output

def parse_passes(passes):

    if not passes.isnumeric() and "/" not in passes:
        return None

    if "/" in passes:
        passes = passes.split("/")

    output = [int(x) for x in passes]
    return output

def parse_tici(tici):
    tici = tici.split("/") if "/" in tici else [tici]
    output = [TICI_MAP[x] for x in tici]
    return output


append_csv_features(get_mips_data())