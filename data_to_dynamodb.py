import numpy as np
import logging
import glob
import os
import numpy as np
import boto3
from sklearn.model_selection import train_test_split
from dynamo import DynamoDBTable
from credentials import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY, AWS_REGION

def split_list_on_element(input_list, split_element='/'):
    """
    Split a list on a specified element and return a list of sublists.
    
    :param input_list: The list to be split.
    :param split_element: The element at which to split the list. Default is '/'.
    :return: A list of sublists after splitting the input list on the specified element.
    """
    result = []
    sublist = []
    for item in input_list:
        if item.find(split_element) > 0:
            if sublist:
                result.append(sublist)
                sublist = [item]
            else:
                sublist.append(item)
        else:
            sublist.append(item)
    if sublist:
        result.append(sublist)
    return result

def map_list_to_json(x):
    """
    Maps a list to a JSON object.

    Args:
        x: The input list containing image_name, num_faces, and faces.

    Returns:
        dict: The JSON object containing image_name, num_faces, faces, and train_set.
    """
    return {
        'image_name': os.path.join(x[0]),
        'num_faces': int(x[1]),
        'faces': x[2:],
        'train_set': 1
    }

# A lambda function that accesses a dictionary key and sets a new value
def modify_value(d):
    """
    Modify the input dictionary by setting the value of 'train_set' to 0.

    Args:
        d: A dictionary to be modified.

    Returns:
        The modified dictionary.
    """
    d['train_set'] = 0
    return d


def run_scenario(table_name, dyn_resource):
    """
    This function runs a scenario with the given table name and DynamoDB resource.
    It checks if the table exists and creates it if it doesn't. 
    It returns the DynamoDBTable object.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dt = DynamoDBTable(dyn_resource)
    table_exists = dt.exists(table_name)

    if not table_exists:
        print(f"\nCreating table {table_name}...")
        dt.create_table(table_name)
        print(f"\nCreated table {dt.table.name}.")
    else:
        print(f"\nTable {table_name} already exists.")

    return dt


if __name__ == "__main__":
    
    # Spcify data paths
    data_src = "temp_data/"
    meta_data_source = "temp_data/FDDB-folds"

    files = np.sort([x for x in glob.glob(meta_data_source + "*/*") if x.find("ellipse") > 0])
    all_lines = []

    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        lines = [x.strip() for x in lines]
        all_lines.append(lines)

    all_sublists = []
    for lines in all_lines:
        sublists = split_list_on_element(lines, split_element = "/")
        all_sublists.append(sublists)
    len(sublists)

    # # Pipe through lambda function
    all_jsons = []
    for sublists in all_sublists:
        jsons = list(map(map_list_to_json, sublists))
        all_jsons.append(jsons)

    # Flatted list of dictionaries
    flat_json_list = [item for sublist in all_jsons for item in sublist]

    # Create train and test sets
    train, test = train_test_split(flat_json_list, test_size=0.2)
    test = list(map(modify_value, test))
    train.extend(test)
    x = np.array(train)

    # Check if duplicates in list
    names = [x['image_name'] for x in x]
    duplicates = [x for x in names if names.count(x) > 1]
    assert len(duplicates) == 0



    dyn = boto3.resource(
        "dynamodb", region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY, 
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    
    dt_name = 'facial-detection-dataset'
    dt = run_scenario(table_name=dt_name, dyn_resource=dyn)

    # Upload data in batches
    dt.write_batch(x)

    print(f"\nDone. {dt.table.name} contains {dt.table.item_count} items.")