
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np
from utility import predict_prob
from sys import argv


def main():
    '''-----
    calling predict function predict_prob(image_path, model_name, top_k) using command line
    use 1 :$ python predict.py /path/to/image saved_model
    use 2 :$ python predict.py /path/to/image saved_model --top_k
    use 3 :$ python predict.py /path/to/image saved_model --category_names map.json
    
    basic usage example:$ python predict.py ./test_images/orchid.jpg my_model.h5
    option 2 usage:$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
    option 3 usage:$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
    ---'''

    # Parse arguments from command line
    my_parser = argparse.ArgumentParser(
        description='This is a flower classification neural network program using pre-trained MobiNet model',
    )
    #my_parser.add_argument('function', type=str, help='function to call')

    my_parser.add_argument('path',
                            type=str, 
                            nargs = 1,
                            help='path to images')

    my_parser.add_argument('model_name',
                            type=str,
                            nargs = 1,
                            help='name of saved trained model')

    my_parser.add_argument("--top_k", 
                            type = int, 
                            default = 5,
                            nargs = 1,
                            help = "specify top k numbers to get probability and classes")

    my_parser.add_argument("--category_names", 
                            default = 'label_map.json',
                            type=str, 
                            nargs = 1,
                            help = "specify json category name file to get the class name")
                            #default='label_map.json'(put this back on later when bug is solved)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    #assigning the command line input as parameter to run the predict_prob funciton.
    print(argv)
    path_image = argv[1]
    my_model = argv[2]

    if len(argv)>=4:
        if argv[3]=="--category_names":
            class_names_file = argv[4]
            k_output = 5
        elif argv[3]=="--top_k":
            k_output = int(argv[4])
            class_names_file ='label_map.json'
        else:
            pass
    else:
        k_output = 5
        class_names_file ='label_map.json'

    #Gettign other parameters to run predict_prob function and get class_names
    model=tf.keras.models.load_model(my_model,custom_objects={'KerasLayer': hub.KerasLayer},compile = False)

    # Call the function using parameters given in command lne argument
    probs, classes=predict_prob(path_image, model, k_output)

    #get the class_names from json file and do mapping to label_names
    #with open(class_names_file, 'r') as json_file:
    with open(class_names_file,'r') as json_file:
        class_names = json.load(json_file)
    label_names = [class_names[str(index+1)] for index in classes]

    #printing results to comand line
    print("top {} probabilities are: {}".format(k_output,probs))
    print("top {} probabilities classes are: {}".format(k_output, classes))
    print("top {} probabilities label names are: {}:".format(k_output, label_names))


if __name__ == '__main__':
    main()

