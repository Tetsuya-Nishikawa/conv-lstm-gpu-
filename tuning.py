from Model import Model
import numpy as np
import tensorflow as tf
import in_data

def search():
    from main import mirrored_strategy
    alpha = [0.0001, 0.01, 0.1]

    #drop_rate = [0, 0.1, 0.2, 0.3, 0.4]
    drop_rate = [0, 0.2]
    drop_menu = []

    i = 0
    for d1 in drop_rate:
        for d2 in drop_rate:
            for d3 in drop_rate:
                for d4 in drop_rate:
                    drop_menu.append([])
                    drop_menu[i] = [d1, d2, d3, d4]
                    i = i + 1

    lambd = [5]
    #alpha : 0 , drop_rate : 0, lambd : 0.1がよかった
    params ={}

    index = 0

    for a in alpha:
        for d in drop_menu:
            for l in lambd:
                params[index] = {}
                params[index]['alpha'] = a
                params[index]['drop_rate'] = d
                params[index]['lambd'] = l
                index = index + 1

    with mirrored_strategy.scope():
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    BATCH_SIZE = 10
    ALPHA = 0.01
    LAMBDA = 3
    EPOCHS = 10
    train_dataset, test_dataset = in_data.read_dataset(BATCH_SIZE)
    train_ds = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    test_ds=   mirrored_strategy.experimental_distribute_dataset(test_dataset)
    #model = Model("Adam", ALPHA, LAMBDA, BATCH_SIZE, train_accuracy, test_accuracy)
    with mirrored_strategy.scope():
        for param in params:
                #ランダムシーケンスリセット
            tf.random.set_seed(seed=1234)

            a = params[param]['alpha']
            d = params[param]['drop_rate']#drop_rateは、全く使用していません。
            l  = params[param]['lambd']

            model = Model("Adam", a, l, BATCH_SIZE, train_accuracy, test_accuracy)

            for epoch in range(EPOCHS):

                mean_loss = 0.0
                test_mean_loss = 0.0
                num_batches = 0.0

             #   if epoch==10:
             #       model.opt.learning_rate = 0.0001
                if (epoch+1)%10==0 :
                    model.opt.learning_rate = model.opt.learning_rate*0.1

                #alpha = (1/(1+(epoch+1)))*0.002
                #hparams[0] = alpha

                for (batch, (train_images, train_labels, masks)) in enumerate(train_ds): 
                    print("hparam : ", params[param], "epoch : ", epoch+1, "batch : ",batch+1)
                    losses             = model.distributed_train_step(train_images, train_labels, masks)
                    mean_loss       = mean_loss +mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None)
                    num_batches += 1.0  
                mean_loss = mean_loss / num_batches
                print("the number of train batch : ", batch+1, "mean_loss : ", mean_loss, "train acc : ", train_accuracy.result().numpy()*100)
                num_batches = 0.0
                for (batch, (test_images, test_labels, masks)) in enumerate(test_ds):
                    losses             = model.distributed_test_step(test_images, test_labels, masks)
                    test_mean_loss  =  test_mean_loss  + mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses,axis=None)
                    num_batches += 1.0

                test_mean_loss = test_mean_loss / num_batches
                print("the number of test batch : ", batch+1)
                print("epoch : ", epoch+1, " | train loss value : ", mean_loss.numpy(), ", test loss value : ", test_mean_loss.numpy())
                print("train acc : ", train_accuracy.result().numpy()*100, "test acc : ", test_accuracy.result().numpy()*100)
                model.accuracy_reset()

