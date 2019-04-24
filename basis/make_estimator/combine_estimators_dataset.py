# blog:https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html

import os
import six.moves.urllib.request as request
import tensorflow as tf

tf_version = tf.__version__
print("Tensorflow version {} ".format(tf_version))
# assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

# 保存数据的位置
PATH = "/media/disk/lds/LDS/DL-tensorflow/tf_dataset_and_estimator_apis"
# Fetch and store Training and Test dataset files
# os.sep 根据你的平台添加分隔符如linux“/”
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep +"iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


def download_dataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.mkdirs(PATH_DATASET)
    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()


download_dataset(URL_TRAIN, FILE_TRAIN)
download_dataset(URL_TEST, FILE_TEST)
tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training & test data
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']

# 使用dataset api 创建一个读取文件的函数
# 然后提供一个返回结果到Estimator API

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
               .skip(1)  # Skip header row
               .map(decode_csv))  # Transform each elem by applying decode_csv fn

    if perform_shuffle:
        # 随机输入使用一个window256个元素（读入memory中）
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


next_batch = my_input_fn(FILE_TRAIN, True) # # Will return 32 random elements
print("nextbatch{}".format(next_batch))
# Create the feature_columns, which specifies the input to our model
# All our input features are numeric, so use numeric_column for each one
feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]
print(feature_columns)

# Create a deep neural network regression classifier
# Use the DNNClassifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  # The input features to our model
    hidden_units=[10, 10],  # Two layers, each with 10 neurons
    n_classes=3,
    model_dir=PATH)  # Path to where checkpoints etc are store
classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
    input_fn=lambda : my_input_fn(FILE_TEST, False, 4)
)
print("Evaluation results")
for key in evaluate_result:
    print("{}, was:{}".format(key, evaluate_result[key]))

predict_results = classifier.predict(
    input_fn=lambda: my_input_fn(FILE_TEST, False, 1))

print("predictions on test file")
for predictions in predict_results:
    print(predictions["class_ids"][0])

# Let create a dataset for prediction
# We've taken the first 3 examples in FILE_TES
prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

def new_input_fn():
    def decode(x):
        x = tf.split(x, 4) # 需要分成4个特征
        print("zip", zip(feature_names, x))
        return  dict(zip(feature_names, x))

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None # # In prediction, we have no labels


predict_results = classifier.predict(input_fn=new_input_fn)
# Print results
print("Predictions:")
for idx, prediction in enumerate(predict_results):
    type = prediction["class_ids"][0]  # Get the predicted class (index)
    if type == 0:
        print("  I think: {}, is Iris Sentosa".format(prediction_input[idx]))
    elif type == 1:
        print("  I think: {}, is Iris Versicolor".format(prediction_input[idx]))
    else:
        print("  I think: {}, is Iris Virginica".format(prediction_input[idx]))
