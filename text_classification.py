#encoding: utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)
import matplotlib
matplotlib.use('Agg')  # 服务器端使用 matplotlib
import matplotlib.pyplot as plt


"""
下载数据集
"""
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data.shape)
print(len(train_labels))
print(test_data.shape)
print(len(test_labels))


"""
查看数据集
"""
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])


"""
制作词典
词：序号
"""
word_index = imdb.get_word_index()
print(word_index)
print(type(word_index))
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
print(len(word_index))


"""
翻转词典
序号：词
"""
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


"""
恢复评论（序号序列转词序列）
"""
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_review(train_data[0]))


"""
padding
"""
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                        maxlen=256)


"""
建模
"""
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


"""
划分出验证集、训练集
"""
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


"""
训练
"""
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=60,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)  # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录


"""
评估模型
"""
results = model.evaluate(test_data, test_labels)
print(results)
history_dict = history.history
print(history_dict.keys())


"""
画图
"""
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  # 图例
plt.savefig("text_classification_loss.png")

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("text_classification_acc.png")
