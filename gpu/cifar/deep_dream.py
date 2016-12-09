model_fn = 'output_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
# imagenet_mean = 117.0
# t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
with graph.as_default():
    tf.import_graph_def(graph_def)