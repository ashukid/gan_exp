import glob
import tensorflow as tf

def get_data(path,image_height=128,image_width=128,image_channel=3,batch_size=64):
    image_path = glob.glob(path+'/*.jpg')
    print("Total Images found : {}".format(len(image_path)))
    Dataset =  tf.data.Dataset
    Iterator = tf.data.Iterator
    
    train_data = Dataset.from_tensor_slices(image_path)
    
    def input_parser(img_path):
        # read the img from file
        img_file = tf.read_file(img_path)
        img_decoded = tf.image.decode_jpeg(img_file, channels=image_channel)
        img_reshaped = tf.image.resize_images(img_decoded,size=(image_height,image_width))

        return img_reshaped
    
    def input_preprocess(img):
        img = tf.divide(img,255)
        img = tf.subtract(tf.multiply(img,2),1) # scaling the image in the tanh range

        return img
    
    train_data = train_data.map(input_parser)
    #train_data = train_data.map(input_preprocess)
    train_data = train_data.repeat()
    print("Creating batches of {}".format(batch_size))
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(1)
    iterator = Iterator.from_structure(train_data.output_types,train_data.output_shapes)
    next_element = iterator.get_next()
    iter_init_op = iterator.make_initializer(train_data)
    
    print("Done reading .. !")
    return next_element,len(image_path),iter_init_op