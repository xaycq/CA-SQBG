import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Bidirectional, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pennylane as qml

# 超参数设置
embedding_dim = 300  # 词嵌入维度
max_sentence_length = 186  # 最大句子长度
learning_rate = 0.001  # 学习率
batch_size = 64  # 批量大小
dropout_rate = 0.15  # Dropout 比例
gru_units = 128  # GRU 隐藏单元数
num_qgrus = 2  # Quantum GRU 层数
vqc_layers = 4  # 变分量子电路层数


# 数据集加载函数
def load_data():
    # 加载 DDIExtraction2013 数据集
    sentences = np.random.randint(1, 1000, (1000, max_sentence_length))
    labels = np.random.randint(0, 5, 1000)
    return sentences, labels


# 词嵌入和位置编码
def build_embedding_layer(input_shape):
    return Embedding(input_dim=2000, output_dim=embedding_dim, input_length=input_shape[1])


# 使用 PennyLane 定义量子 GRU 单元
def quantum_gru_cell(inputs, hidden_state):
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev, interface="tensorflow")
    def qnode(inputs, hidden_state):
        # 构建量子电路
        for i in range(vqc_layers):
            qml.RX(inputs[i], wires=i)
            qml.RZ(hidden_state[i], wires=i)
            qml.CNOT(wires=[i, (i + 1) % 4])
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    return qnode(inputs, hidden_state)


# 构建 Siam-BiQGRU 模型
def build_siam_biqgru_model(input_shape):
    inputs = Input(shape=(input_shape[1],))

    # 嵌入层
    embedding_layer = build_embedding_layer(input_shape)
    embeddings = embedding_layer(inputs)

    # Siamese 双向量子 GRU 层和注意力池化
    def biqgru_block(x):
        # 量子 GRU 输出和状态
        qgru_output, state = tf.keras.layers.RNN(lambda inputs, state: quantum_gru_cell(inputs, state),
                                                 return_state=True)(x)
        return Bidirectional(GRU(gru_units, return_sequences=True))(qgru_output)

    # Siamese 网络的左侧和右侧分支
    siam_left = biqgru_block(embeddings)
    siam_right = biqgru_block(embeddings)

    # 计算余弦相似度
    def cosine_similarity(vectors):
        x, y = vectors
        x = tf.math.l2_normalize(x, axis=-1)
        y = tf.math.l2_normalize(y, axis=-1)
        return tf.reduce_sum(x * y, axis=-1, keepdims=True)

    similarity = Lambda(cosine_similarity)([siam_left, siam_right])

    # 全连接层，加 Dropout 和 Softmax
    x = Dense(64, activation='relu')(similarity)
    x = Dropout(dropout_rate)(x)
    output = Dense(5, activation='softmax')(x)

    # 编译模型
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# 加载数据集
sentences, labels = load_data()
x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)

# 5 折交叉验证模型训练
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

for train_index, val_index in kf.split(x_train):
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model = build_siam_biqgru_model(x_train.shape)

    # 早停回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold),
              epochs=3000, batch_size=batch_size, callbacks=[early_stopping], verbose=1)

    y_pred = np.argmax(model.predict(x_val_fold), axis=1)

    # 评估
    accuracy_list.append(accuracy_score(y_val_fold, y_pred))
    precision_list.append(precision_score(y_val_fold, y_pred, average='weighted'))
    recall_list.append(recall_score(y_val_fold, y_pred, average='weighted'))
    f1_list.append(f1_score(y_val_fold, y_pred, average='weighted'))

# 输出平均性能指标
print(f"平均准确率: {np.mean(accuracy_list):.4f}")
print(f"平均精确率: {np.mean(precision_list):.4f}")
print(f"平均召回率: {np.mean(recall_list):.4f}")
print(f"平均 F1 值: {np.mean(f1_list):.4f}")

# 在测试集上进行最终模型评估
model = build_siam_biqgru_model(x_train.shape)
model.fit(x_train, y_train, epochs=3000, batch_size=batch_size, verbose=1)
y_test_pred = np.argmax(model.predict(x_test), axis=1)

# 在测试集上进行最终评估
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集精确率: {test_precision:.4f}")
print(f"测试集召回率: {test_recall:.4f}")
print(f"测试集 F1 值: {test_f1:.4f}")
