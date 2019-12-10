#!/usr/bin/python3
# 2019.11.13
# Author Zhang Yihao @NUS

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Lambda, Reshape
from keras.layers.core import Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from evaluate import evaluate_model
from time import time
import argparse
from Dataset_topN import Dataset

# ============Arguments ==================
# --k 和--num_factors 两个参数必须相等
factor = 5
dataName = 'Baby'
topN = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Run ReSys_TopN.")
    parser.add_argument('--path', nargs='?', default='Data/' + dataName + '/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=dataName,
                        help='Choose a dataset.')
    parser.add_argument('--k', type=int, default=factor,
                        help='Number of attention factor')
    parser.add_argument('--activation_function', nargs='?', default='relu',
                        help='activation functions')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=factor,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='Learning rate.')
    parser.add_argument('--latent_layer_dim', nargs='?', default='[40, 20, 10]',
                        help="Embedding size for each layer")
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def init_normal(shape):
    return K.random_normal(shape, mean=0, stddev=0.01, seed=None)


def user_attention(user_latent, user_atten):
    latent_size = user_latent.shape[1].value

    inputs = Concatenate()([user_latent, user_atten])
    output = Dense(latent_size,
                   activation='relu',
                   kernel_initializer='glorot_normal',
                   kernel_regularizer=l2(0.001),
                   name='user_attention_layer')(inputs)
    latent = Lambda(lambda x: K.softmax(x), name='user_attention_softmax')(output)
    output = Multiply()([user_latent, latent])
    return output


def item_attention(item_latent, item_atten):
    latent_size = item_latent.shape[1].value

    inputs = Concatenate()([item_latent, item_atten])
    output = Dense(latent_size,
                   activation='relu',
                   kernel_initializer='glorot_normal',
                   kernel_regularizer=l2(0.001),
                   name='item_attention_layer')(inputs)
    latent = Lambda(lambda x: K.softmax(x), name='item_attention_softmax')(output)
    output = Multiply()([item_latent, latent])
    return output


def get_model(num_users, num_items, k, layers, latent_dim, regs=[0, 0]):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_atten = Input(shape=(k,), dtype='float32', name='user_attention')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    item_atten = Input(shape=(k,), dtype='float32', name='item_attention')

    Embedding_User = Embedding(input_dim=num_users,
                                  input_length=1,
                                  output_dim=latent_dim,
                                  embeddings_initializer=init_normal,
                                  embeddings_regularizer=l2(regs[0]),
                                  name='user_embedding')

    Embedding_Item = Embedding(input_dim=num_items,
                                  input_length=1,
                                  output_dim=latent_dim,
                                  embeddings_initializer=init_normal,
                                  embeddings_regularizer=l2(regs[0]),
                                  name='item_embedding')

    # Crucial to flatten an embedding vector!
    user_latent = Reshape((latent_dim,))(Flatten()(Embedding_User(user_input)))
    item_latent = Reshape((latent_dim,))(Flatten()(Embedding_Item(item_input)))
    mul_vector = Multiply()([user_latent, item_latent])

    # user_latent = Concatenate()([user_fea, user_latent])
    # item_latent = Concatenate()([item_fea, item_latent])
    user_latent_atten = user_attention(user_latent, user_atten)
    item_latent_atten = item_attention(item_latent, item_atten)
    con_vector = Concatenate()([user_latent_atten, item_latent_atten])
    # user_latent
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation='relu',
                      name='user_layer%d' % idx)
        mul_vector = layer(mul_vector)

    # item_latent
    for idx in range(0, len(layers)):
        layer = Dense(layers[idx],
                      kernel_regularizer=l2(0.001),
                      kernel_initializer='glorot_normal',
                      activation='relu',
                      name='item_layer%d' % idx)
        con_vector = layer(con_vector)

    # vec = Multiply()([user_latent, item_latent])
    user_item_multi = Multiply()([mul_vector, con_vector])

    prediction_layer = Dense(1,
                             activation='sigmoid',
                             kernel_initializer='lecun_normal',
                             name='prediction')
    # att = prediction_layer(user_item_concat)
    prediction = prediction_layer(user_item_multi)
    ae_model = Model(inputs=[user_input, user_atten, item_input, item_atten], outputs=prediction)
    return ae_model


def get_train_instances(train, user_review_fea, item_review_fea):
    user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
    num_user = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        user_fea.append(user_review_fea[u])
        item_input.append(i)
        item_fea.append(item_review_fea[i])
        label = train[u, i]
        labels.append(label)
    # one_hot_labels = keras.utils.to_categorical(labels, num_classes=5)
    return np.array(user_input), np.array(user_fea, dtype='float32'), np.array(item_input), \
           np.array(item_fea, dtype='float32'), np.array(labels)


if __name__ == '__main__':
    args = parse_args()
    num_factors = args.num_factors
    k = args.k
    layers = eval(args.latent_layer_dim)
    regs = eval(args.regs)
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose
    activation_function = args.activation_function


    evaluation_threads = 1  # mp.cpu_count()
    print("ReSys_TopN: %s" % topN)
    print("ReSys_NCRAE arguments: %s" % args)
    model_out_file = 'modelSave/%sNumofTopic_%d_GMF_%d_%d.h5' % (args.dataset, k, num_factors, time())

    # Loading data
    t1 = time()

    dataset = Dataset(args.path + args.dataset, k)
    train, user_review_fea, item_review_fea, testRatings, testNegatives = dataset.trainMatrix, dataset.user_review_fea, \
                                                                          dataset.item_review_fea, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, k, layers, num_factors, regs)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss="binary_crossentropy")
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss="binary_crossentropy")
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss="binary_crossentropy")
    # print(model.summary())

    # Init performance
    t1 = time()

    (hits, ndcgs) = evaluate_model(model, train, testRatings, testNegatives, user_review_fea, item_review_fea, topN,
                                   evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1

    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, user_fea, item_input, item_fea, labels = get_train_instances(train, user_review_fea,
                                                                                 item_review_fea)
        # Training the model
        hist = model.fit([user_input, user_fea, item_input, item_fea],  # input
                         labels,  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation the model
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model(model, train, testRatings, testNegatives, user_review_fea, item_review_fea,
                                           topN, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR@10 = %.4f, NDCG@10 = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best cm_topN model is saved to %s" % (model_out_file))
