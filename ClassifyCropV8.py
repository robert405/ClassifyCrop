from Utils import *
import glob
import tensorflow as tf
import os

globPoolCount = 0
#tensorflow
def weight_variable(shape, vName):

    var = tf.get_variable(vName, shape, initializer=tf.contrib.layers.xavier_initializer())
    return var

def bias_variable(shape, vName):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name = vName)

def conv2d(x, W):

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x, ss):

    global globPoolCount
    globPoolCount += 1
    poolname = "pool" + str(ss) + "x-" + str(globPoolCount)

    with tf.name_scope(poolname):
        h_pool = tf.nn.max_pool(x, ksize=[1, ss, ss, 1], strides=[1, ss, ss, 1], padding='SAME', name=poolname)

    return h_pool

def convPoolLayer(nbLayer, inputScore, kernelSize, kernelDeep, outputSize, varDict):

    with tf.name_scope("convpool-"  + str(nbLayer)):
        Wname = "W_conv" + str(nbLayer)
        bname = "b_conv" + str(nbLayer)

        W_conv = weight_variable([kernelSize, kernelSize, kernelDeep, outputSize], Wname)
        b_conv = bias_variable([outputSize], bname)

        varDict[Wname] = W_conv
        varDict[bname] = b_conv

        score = tf.add(tf.nn.conv2d(inputScore, W_conv, strides=[1, 2, 2, 1], padding='SAME'), b_conv)
        relu = tf.maximum(0.01 * score, score, name=("convpoolRelu" + str(nbLayer)))

        tf.summary.histogram("histogram-" + Wname, W_conv)
        tf.summary.histogram("histogram-" + bname, b_conv)

    return relu


def convLayer(nbLayer, inputScore, kernelDeep, outputSize, varDict):

    with tf.name_scope("conv-"  + str(nbLayer)):
        Wname = "W_conv" + str(nbLayer)
        bname = "b_conv" + str(nbLayer)

        W_conv = weight_variable([3, 3, kernelDeep, outputSize], Wname)
        b_conv = bias_variable([outputSize], bname)

        varDict[Wname] = W_conv
        varDict[bname] = b_conv

        score = conv2d(inputScore, W_conv) + b_conv

        tf.summary.histogram("histogram-" + Wname, W_conv)
        tf.summary.histogram("histogram-" + bname, b_conv)

    return score

def fullyLayer(nbLayer, inputScore, inputSize, outputSize, varDict):

    with tf.name_scope("fc-" + str(nbLayer)):
        Wname = "W_fc" + str(nbLayer)
        bname = "b_fc" + str(nbLayer)
        W_fc = weight_variable([inputSize, outputSize], Wname)
        b_fc = bias_variable([outputSize], bname)

        varDict[Wname] = W_fc
        varDict[bname] = b_fc

        score = tf.matmul(inputScore, W_fc) + b_fc
        h_fc = tf.maximum(0.01 * score, score, name=("fcRelu" + str(nbLayer)))

        tf.summary.histogram("histogram-" + Wname, W_fc)
        tf.summary.histogram("histogram-" + bname, b_fc)

    return h_fc

def resLayer(nbLayer, inputx, inputSize, outputSize, varDict):

    with tf.name_scope("resLayer-" + str(nbLayer)):

        residu = inputx

        if (outputSize == (2 * inputSize)):
            residu = tf.concat([inputx, inputx], 3)

        score1 = convLayer(str(nbLayer + 0.1), inputx, inputSize, inputSize, varDict)
        relu1 = tf.maximum(0.01 * score1, score1, name=("resRelu" + str(nbLayer + 0.1)))
        score2 = convLayer(str(nbLayer + 0.2), relu1, inputSize, outputSize, varDict)
        score3 = tf.add(score2, residu)
        relu2 = tf.maximum(0.01 * score3, score3, name=("resRelu" + str(nbLayer + 0.2)))

    return relu2

sess = tf.Session()
variableDict = {}

x = tf.placeholder(tf.float32, shape=[None, 224,224,3], name = "x")
y_ = tf.placeholder(tf.float32, shape=[None], name = "y_")

#-------------------------------------------------------------

h_conv = convPoolLayer(1, x, 7, 3, 64, variableDict)

# 1
h_conv = max_pool(h_conv, 2)

h_conv = resLayer(1, h_conv, 64, 128, variableDict)

# 2
h_conv = max_pool(h_conv, 2)

h_conv = resLayer(2, h_conv, 128, 256, variableDict)

# 3
h_conv = max_pool(h_conv, 2)

h_conv = resLayer(3, h_conv, 256, 512, variableDict)

# 4
h_conv = max_pool(h_conv, 2)

h_conv = resLayer(4, h_conv, 512, 512, variableDict)

h_conv = resLayer(5, h_conv, 512, 512, variableDict)

#------------------------------------------------------------

# global average pooling
h_avPool = tf.layers.average_pooling2d(h_conv, [7, 7], [7, 7])
inputSize = 1 * 1 * 512
h_avPool_flat = tf.reshape(h_avPool, [-1, inputSize])

#------------------------------------------------------------


# reasout layer (softmax)
with tf.name_scope("softmax"):
    W_fcs = weight_variable([512, 2], "W_fcs")
    b_fcs = bias_variable([2], "b_fcs")

    variableDict["W_fcs"] = W_fcs
    variableDict["b_fcs"] = b_fcs

    y_conv = tf.add(tf.matmul(h_avPool_flat, W_fcs), b_fcs, name = "y_conv")

    tf.summary.histogram("histogram-W_fcs", W_fcs)
    tf.summary.histogram("histogram-b_fcs", b_fcs)

#------------------------------------------------------------
general_model_path = "/home/martin/Desktop/Free Dev/ModelSave/"
# 'Saver' op to save and restore all the variables

learning_rate = tf.placeholder(tf.float32, shape=[], name = "learning_rate")

with tf.name_scope("trainStep"):
    onehot_labels = tf.one_hot(indices=tf.cast(y_, tf.int32), depth=2)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(onehot_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/martin/Desktop/Free Dev/TensorBoard")
writer.add_graph(sess.graph)
init = tf.global_variables_initializer()
saver = tf.train.Saver(variableDict)
sess.run(init)

goodPath = "/home/martin/Desktop/Free Dev/Data/good/*.jpg"
badPath = "/home/martin/Desktop/Free Dev/Data/bad/*.jpg"
allGoodFile = glob.glob(goodPath)
allBadFile = glob.glob(badPath)

iteration = (int)(len(allGoodFile))
if (len(allGoodFile) > len(allBadFile)):
    iteration = (int)(len(allBadFile))

shuf = np.arange(iteration)
np.random.shuffle(shuf)

lr = 1e-4
iteration = iteration * 2
print("Going for : " + str(iteration) + " iterations")
print("---------------------------------------------")

for i in range(0, iteration, 1):

    i1 = shuf[(i % len(shuf))]
    #i2 = shuf[i+1]
    goodFiles = [allGoodFile[i1]]
    badFiles = [allBadFile[i1]]
    data, label = getData(goodFiles, badFiles, True, True)

    if (i % 50 == 0):

        print("Training step " + str(i))
        print("Learning rate : " + str(lr))
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: data, y_: label, learning_rate:lr})
        print("accuracy %g" % (train_accuracy))
        print("---------------------------------------------")

    if (i % 8000 == 0 and i != 0):
        lr = lr * 0.1

    summary, _ = sess.run([merged, train_step], feed_dict={x: data, y_: label, learning_rate:lr})
    writer.add_summary(summary, i)


goodTestPath = "/home/martin/Desktop/Free Dev/Data/goodTest/*.jpg"
badTestPath = "/home/martin/Desktop/Free Dev/Data/badTest/*.jpg"
goodTestFile = glob.glob(goodTestPath)
badTestFile = glob.glob(badTestPath)

testData, testLabel = getData(goodTestFile, badTestFile, False, False)

train_accuracy = accuracy.eval(session=sess, feed_dict={x: testData, y_: testLabel, learning_rate:lr})
print("accuracy %g"%(train_accuracy))

# Save model weights to disk
model_path = general_model_path + "model-It" + str(iteration) + "-Acc" + str((int)(train_accuracy * 100))
os.makedirs(model_path)
save_path = saver.save(sess, model_path + "/model.ckpt")
print("Model saved in file: %s" % save_path)
