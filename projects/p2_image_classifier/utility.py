from PIL import Image
import tensorflow as tf
import numpy as np

def predict_prob(image_path, model, top_k):

    #print(image_path)
    image = Image.open(image_path)
    test_image = np.asarray(image)

    processed_test_image = process_image(test_image)

    converted_test_image = np.expand_dims(processed_test_image, axis=0)
    #below line does the same thing but above line is shorter and more efficient
    #converted_test_image = processed_test_image.reshape(1,processed_test_image.shape[0],processed_test_image.shape[1],processed_test_image.shape[2])
    #print("shape of the converted image array is:{}".format(converted_test_image.shape))

    prob_preds= model.predict(converted_test_image)[0]
    
    #predicted_classes = model.predict_classes(converted_test_image, batch_size=batch_size, verbose=0)

    values, indices = tf.math.top_k(prob_preds, k=tf.squeeze(top_k))
    probs = values.numpy().tolist()
    classes = indices.numpy().tolist()

    #print(probs)
    #print(classes)
    
    return probs, classes

def process_image(image):
    image_size = 224
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    
    return image