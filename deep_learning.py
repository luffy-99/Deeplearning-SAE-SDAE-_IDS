import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import shuttle
import operator
import os.path
import classifier
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import time
import chartcolumn
import pandas

hidden_layers = np.asarray([[150, 100, 50], [30, 20, 10], [1000, 500, 100],
                            [26, 15, 10], [80,50,12], [42, 27, 9],[450,200,100],[400,300,100],[85, 30, 15]])
batch_sizes = np.asarray([100, 100, 100, 100, 100, 100, 100, 100,100])

def make_init_deep_learning(data_index, pathTrain, pathTest, pathColumn,AUC_and_Structure):
    datasets = np.asarray(["unsw", "ctu13_8", "Ads", "Phishing", "IoT", "Spam", "Antivirus", "VirusShare",'nslkdd'])#30 ? 57 513 482

    dataname = datasets[data_index]
    dt = shuttle.read_data_sets(dataname, pathTrain, pathTest, pathColumn)
    num_sample = dt.train.num_examples
    chartcolumn.take_display_traning("size of dataset: " + str(num_sample))
    input_dim = dt.train.features.shape[1]
    balance_rate = len(dt.train.features1) / float(len(dt.train.features0))

    label_dim = dt.train.labels.shape[1]
    chartcolumn.take_display_traning("dimension: " + str(input_dim))
    chartcolumn.take_display_traning("number of class: " + str(label_dim))

    data_save = np.asarray([data_index, input_dim, balance_rate, label_dim])
    data_save = np.reshape(data_save, (-1, 4))
    if os.path.isfile("Results/Infomation/" + dataname + "/datainformation.csv"):  #
        auc = np.genfromtxt("Results/Infomation/" + dataname + "/datainformation.csv", delimiter=',')
        auc = np.reshape(auc, (-1, 4))
        data_save = np.concatenate((auc, data_save), axis=0)
        np.savetxt("Results/Infomation/" + dataname + "/datainformation.csv", data_save, delimiter=",", fmt="%f")

    else:
        np.savetxt("Results/Infomation/" + dataname + "/datainformation.csv", data_save, delimiter=",", fmt="%f")

    num_epoch = 2000
    step = 20
   # filter_sizes = np.asarray([[1, 2, 3], [3, 5, 7], [1, 2, 3], [7, 11, 15], [1, 2, 3], [3, 5, 7], [1, 2, 3]])
    #data_shapes = np.asarray([[12, 12], [14, 14], [8, 8], [40, 40], [9, 9], [25, 25], [8, 8]])
    hidden_layer = hidden_layers[data_index]
    block_size = batch_sizes[data_index]
    lr = 1e-4
    noise_factor = 0.0025  # 0, 0.0001, 0,001, 0.01, 0.1, 1.0
   # filter_size = filter_sizes[data_index]  # [1,2,3]
    #data_shape = data_shapes[data_index]  # [12,12]
    conf = str(num_epoch) + "_" + str(block_size) + "_" + str(lr) + "_" + str(hidden_layer[0]) + "_" + str(
        hidden_layer[1]) + "_" + str(hidden_layer[2]) + "_noise: " + str(noise_factor)
    
    X_train = dt.train.features
    Y_train = dt.train.labels
    X_test = dt.test.features
    Y_test = dt.test.labels

    svm, t1 = classifier.svm(X_train, Y_train, X_test, Y_test)
    auc_dt, t2 = classifier.decisiontree(X_train, Y_train, X_test, Y_test)
    rf, t3 = classifier.rf(X_train, Y_train, X_test, Y_test)
    nb, t4 = classifier.naive_baves(X_train, Y_train, X_test, Y_test)
    kn, t5 = classifier.KNeighbors(X_train, Y_train, X_test, Y_test)
    logistic, t6 = classifier.Logistic(X_train, Y_train, X_test, Y_test)
    
    data_save = np.asarray(
        [data_index, input_dim, balance_rate, svm, auc_dt, rf, nb, 1000 * t1, 1000 * t2, 1000 * t3, 1000 * t4])
    data_save = np.reshape(data_save, (-1, 11))
    AUC_and_Structure.append(svm)
    AUC_and_Structure.append(auc_dt)
    AUC_and_Structure.append(rf)
    AUC_and_Structure.append(nb)
    AUC_and_Structure.append(kn)
    AUC_and_Structure.append(logistic)
    if os.path.isfile("Results/RF_AUC_DIF/" + dataname + "/AUC_Input.csv"):  #
        auc = np.genfromtxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Input.csv", delimiter=',')
        auc = np.reshape(auc, (-1, 11))
        data_save = np.concatenate((auc, data_save), axis=0)
        np.savetxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Input.csv", data_save, delimiter=",", fmt="%f")

    else:
        np.savetxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Input.csv", data_save, delimiter=",", fmt="%f")
    return data_index, input_dim, balance_rate, lr, block_size, num_epoch, hidden_layer, \
           step, X_train, X_test, Y_train, Y_test, dt, label_dim, noise_factor, conf, \
           dataname


def save_auc_tofile(data_index, input_dim, balance_rate, method, svms, dts, rfs, nab, kn, logistic, t1s, t2s, t3s, t4s,t5s,t6s, dataname,AUC_and_Structure):
    index1, svm = max(enumerate(svms), key=operator.itemgetter(1))
    index2, dt = max(enumerate(dts), key=operator.itemgetter(1))
    index3, rf = max(enumerate(rfs), key=operator.itemgetter(1))
    index4, nab = max(enumerate(nab), key=operator.itemgetter(1))
    index5, kn = max(enumerate(kn), key=operator.itemgetter(1))
    index6, logistic = max(enumerate(logistic), key=operator.itemgetter(1))
    
    t1 = 1000 * sum(t1s) / len(t1s)
    t2 = 1000 * sum(t2s) / len(t2s)
    t3 = 1000 * sum(t3s) / len(t3s)
    t4 = 1000 * sum(t4s) / len(t4s)
    t5 = 1000 * sum(t5s) / len(t5s)
    t6 = 1000 * sum(t6s) / len(t6s)
    AUC_and_Structure.append(svm)
    AUC_and_Structure.append(dt)
    AUC_and_Structure.append(rf)
    AUC_and_Structure.append(nab)
    AUC_and_Structure.append(kn)
    AUC_and_Structure.append(logistic)
    AUC_and_Structure.append(method)
    print(AUC_and_Structure)
    dt = pandas.read_csv("Results/RF_AUC_DIF/" + dataname +"/hiddenlayer.csv")
    dt = dt.drop('Unnamed: 0', axis =1)
    data = pandas.DataFrame([AUC_and_Structure], columns=['layer1','layer2','layer3',
                                                 'svm','dt','rf','nb','logistic','kn','ae_svm','ae_dt','ae_rf','ae_nb','ae_kn','ae_logistic','method'])

    dt = dt.append(data, sort=False, ignore_index=True)
    dt.to_csv("Results/RF_AUC_DIF/" + dataname + "/hiddenlayer.csv")
'''    data_save = np.asarray([data_index, input_dim, balance_rate, svm, dt, rf, nab,kn,logistic, t1, t2, t3, t4,t5,t6])
    data_save = np.reshape(data_save, (-1, 15))
    
    if os.path.isfile("Results/RF_AUC_DIF/" + dataname + "/AUC_Hidden_" + method + ".csv"):  #
        auc = np.genfromtxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Hidden_" + method + ".csv", delimiter=',')
        auc = np.reshape(auc, (-1, 15))
        data_save = np.concatenate((auc, data_save), axis=0)
        np.savetxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Hidden_" + method + ".csv", data_save, delimiter=",",fmt="%f")

    else:
        np.savetxt("Results/RF_AUC_DIF/" + dataname + "/AUC_Hidden_" + method + ".csv", data_save, delimiter=",",fmt="%f")
    return 1'''
   


# -----------------------------------------------------------------------------------------------------------

class AE():
    def __init__(self, learning_rate=1e-3, batch_size=100, hidden_layers=[85, 30, 12], input_dim=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.build(input_dim)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self, input_dim):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, self.hidden_layers[0], scope='ae_enc_fc1', activation_fn=tf.nn.relu)
        # f2 = fc(f1, 60, scope='enc_fc2', activation_fn=tf.nn.tanh)
        f3 = fc(f1, self.hidden_layers[1], scope='ae_enc_fc3', activation_fn=tf.nn.relu)
        # f4 = fc(f3, 20, scope='enc_fc4', activation_fn=tf.nn.relu)

        self.z = fc(f3, self.hidden_layers[2], scope='ae_enc_fc5_mu', activation_fn=None)

        # Decode
        # z,y -> x_hat
        # g1 = fc(self.Z, 20, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(self.z, self.hidden_layers[1], scope='ae_dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, self.hidden_layers[0], scope='ae_dec_fc3', activation_fn=tf.nn.relu)
        # g4 = fc(g3, 85, scope='dec_fc4', activation_fn=tf.nn.tanh)

        self.x_hat = fc(g3, input_dim, scope='ae_dec_fc5', activation_fn=tf.sigmoid)
        # self.x_res = self.x_hat[:,0:input_dim]

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        recon_loss = tf.reduce_mean(tf.square(self.x - self.x_hat), 1)  # (((self.x - y)**2).mean(1)).mean()
        # epsilon = 1e-10
        # recon_loss = -tf.reduce_sum(
        #    self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
        #    axis=1
        # )
        self.recon_loss = tf.reduce_mean(recon_loss)

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.recon_loss)

        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, recon_loss = self.sess.run(
            [self.train_op, self.recon_loss],
            feed_dict={self.x: x}

        )

        return recon_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


AE_recon_loss_ = []
AE_auc_svm_ = []
AE_auc_dt_ = []
AE_auc_rf_ = []
AE_auc_nb_ = []
AE_auc_kn_ = []
AE_auc_logistic_ = []
AE_t1_ = []
AE_t2_ = []
AE_t3_ = []
AE_t4_ = []
AE_t5_ = []
AE_t6_ = []

def AE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers=[7, 4, 2],
               input_dim=0, step=20, X_train=[], X_test=[], Y_train=[], Y_test=[], dt=[]):
    model1 = AE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers, input_dim=input_dim)
    for epoch in range(num_epoch):

        num_sample = len(X_train)
        for iter in range(num_sample // batch_size):
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss = model1.run_single_step(X_mb)

        if epoch % step == 0:
            print('[Epoch {}] Recon loss: {}'.format(
                epoch, recon_loss))
            chartcolumn.take_display_traning('Epoch ' + str(epoch) + ' Recon loss: ' + str(recon_loss))
            # model.writer.add_summary(summary, epoch )

            z_train = model1.transformer(X_train)
            s = time.time()
            z_test = model1.transformer(X_test)
            e = time.time()
            t_tr = (e - s) / float(len(X_test))
            # np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            # np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            auc_svm, t1 = classifier.svm(z_train, Y_train, z_test, Y_test)
            auc_dt, t2 = classifier.decisiontree(z_train, Y_train, z_test, Y_test)
            auc_rf, t3 = classifier.rf(z_train, Y_train, z_test, Y_test)
            auc_nb, t4 = classifier.naive_baves(z_train, Y_train, z_test, Y_test)
            auc_kn, t5 = classifier.KNeighbors(z_train, Y_train, z_test, Y_test)
            auc_logistic, t6 = classifier.Logistic(z_train, Y_train, z_test, Y_test)
            AE_recon_loss_.append(recon_loss)

            AE_auc_svm_.append(auc_svm)
            AE_auc_dt_.append(auc_dt)
            AE_auc_rf_.append(auc_rf)
            AE_auc_nb_.append(auc_nb)
            AE_auc_kn_.append(auc_kn)
            AE_auc_logistic_.append(auc_logistic)

            AE_t1_.append((t1 + t_tr))
            AE_t2_.append((t2 + t_tr))
            AE_t3_.append((t3 + t_tr))
            AE_t4_.append((t4 + t_tr))
            AE_t5_.append((t5 + t_tr))
            AE_t6_.append((t6 + t_tr))

    print('Done AE!')

    return model1


def call_AE_train(data_index, input_dim, balance_rate, lr, block_size, num_epoch, hidden_layer,
                  step, X_train, X_test, Y_train, Y_test, dt, dataname, conf, algorithm, AUC_and_Structure):
    print(X_train)
    model = AE_trainer(learning_rate=lr, batch_size=block_size, num_epoch=num_epoch,
                       hidden_layers=hidden_layer, input_dim=input_dim, step=step,
                       X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, dt=dt)
    # save auc to file:
    save_auc_tofile(data_index, input_dim, balance_rate, "AE", AE_auc_svm_, AE_auc_dt_, AE_auc_rf_, AE_auc_nb_, AE_auc_kn_, AE_auc_logistic_, AE_t1_, AE_t2_, AE_t3_,AE_t4_, AE_t5_, AE_t6_, dataname, AUC_and_Structure)
    header = "epoch, ae_loss, ae_svm, ae_dt, ae_rf, ae_nb, ae_kn, ae_logistic, ae_svm_time, ae_dt_time, ae_rf_time, ae_nb_time, ae_kn_time, ae_logistic_time"
    '''write_fie(num_epoch, step, dataname, conf, header, [], AE_recon_loss_, [], AE_auc_svm_, AE_auc_dt_, AE_auc_rf_, AE_auc_nb_,AE_t1_, AE_t2_, AE_t3_,AE_t4_, "AE", algorithm)'''


# ----------------------------------------------------------------------------------------

class VAE(object):

    def __init__(self, learning_rate=1e-3, batch_size=100, hidden_layers=[85, 30, 12], input_dim=0):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.build(input_dim)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self, input_dim):
        print(input_dim)
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])

        # Encode
        # x -> z_mean, z_sigma -> z
        f1 = fc(self.x, self.hidden_layers[0], scope='vae_enc_fc1', activation_fn=tf.nn.relu)
        # f2 = fc(f1, 60, scope='enc_fc2', activation_fn=tf.nn.tanh)
        f3 = fc(f1, self.hidden_layers[1], scope='vae_enc_fc3', activation_fn=tf.nn.relu)
        # f4 = fc(f3, 20, scope='enc_fc4', activation_fn=tf.nn.relu)

        self.z_mu = fc(f3, self.hidden_layers[2], scope='vae_enc_fc5_mu', activation_fn=None)
        self.z_log_sigma_sq = fc(f3, self.hidden_layers[2], scope='vae_enc_fc5_sigma', activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),
                               mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        # Decode
        # z,y -> x_hat
        # g1 = fc(self.Z, 20, scope='dec_fc1', activation_fn=tf.nn.relu)
        g2 = fc(self.z, self.hidden_layers[1], scope='vae_dec_fc2', activation_fn=tf.nn.relu)
        g3 = fc(g2, self.hidden_layers[0], scope='vae_dec_fc3', activation_fn=tf.nn.relu)
        # g4 = fc(g3, 85, scope='dec_fc4', activation_fn=tf.nn.tanh)

        self.x_hat = fc(g3, input_dim, scope='vae_dec_fc5', activation_fn=tf.sigmoid)
        # self.x_res = self.x_hat[:,0:input_dim]

        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        recon_loss = tf.reduce_mean(tf.square(self.x - self.x_hat), 1)  # (((self.x - y)**2).mean(1)).mean()
        # epsilon = 1e-10
        # recon_loss = -tf.reduce_sum(
        #    self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+1-self.x_hat),
        #    axis=1
        # )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)

        # original
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)

        self.latent_loss = tf.reduce_mean(latent_loss)
        self.total_loss = tf.reduce_mean(recon_loss + latent_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}

        )

        return loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


VAE_loss_ = []
VAE_recon_loss_ = []
VAE_latent_loss_ = []
VAE_auc_svm_ = []
VAE_auc_dt_ = []
VAE_auc_rf_ = []
VAE_auc_nb_ = []
VAE_t1_ = []
VAE_t2_ = []
VAE_t3_ = []
VAE_t4_ = []


def VAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers=[7, 4, 2],
                input_dim=0, step=20, X_train=[], X_test=[], Y_train=[], Y_test=[], dt=[]):
    model = VAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers, input_dim=input_dim)
    for epoch in range(num_epoch):

        num_sample = len(X_train)
        for iter in range(num_sample // batch_size):
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            loss, recon_loss, latent_loss = model.run_single_step(X_mb)

        if epoch % step == 0:
            print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
                epoch, loss, recon_loss, latent_loss))
            chartcolumn.take_display_traning('Epoch ' + str(epoch) + ' Loss: ' + str(loss) + ' Recon loss: ' + str(
                recon_loss) + ' Latent loss: ' + str(latent_loss))

            z_train = model.transformer(X_train)
            s = time.time()
            z_test = model.transformer(X_test)
            e = time.time()
            t_tr = (e - s) / len(X_test)
            # np.savetxt(path +  "z_train_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            # np.savetxt(path +  "z_test_"+str(epoch)+".csv", z_train, delimiter=",", fmt='%f' )
            auc_svm, t1 = classifier.svm(z_train, Y_train, z_test, Y_test)
            auc_dt, t2 = classifier.decisiontree(z_train, Y_train, z_test, Y_test)
            auc_rf, t3 = classifier.rf(z_train, Y_train, z_test, Y_test)
            auc_nb, t4 = classifier.naive_baves(z_train, Y_train, z_test, Y_test)

            VAE_loss_.append(loss)
            VAE_recon_loss_.append(recon_loss)
            VAE_latent_loss_.append(latent_loss)

            VAE_auc_svm_.append(auc_svm)
            VAE_auc_dt_.append(auc_dt)
            VAE_auc_rf_.append(auc_rf)
            VAE_auc_nb_.append(auc_nb)
            VAE_t1_.append((t1 + t_tr))
            VAE_t2_.append((t2 + t_tr))
            VAE_t3_.append((t3 + t_tr))
            VAE_t4_.append((t4 + t_tr))

    print('Done VAE!')

    return model


def call_VAE_train(data_index, input_dim, balance_rate, lr, block_size, num_epoch, hidden_layer,
                   step, X_train, X_test, Y_train, Y_test, dt, dataname, conf, algorithm):
    model = VAE_trainer(learning_rate=lr, batch_size=block_size, num_epoch=num_epoch,
                        hidden_layers=hidden_layer, input_dim=input_dim, step=step,
                        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, dt=dt)
    # save auc to file:
    save_auc_tofile(data_index, input_dim, balance_rate, "VAE", VAE_auc_svm_, VAE_auc_dt_, VAE_auc_rf_, VAE_auc_nb_,
                    VAE_t1_, VAE_t2_, VAE_t3_, VAE_t4_, dataname)
    header = "epoch, vae_loss, vae_RE, vae_KL, vae_svm, vae_dt, vae_rf, vae_nb, vae_svm_time, vae_dt_time, vae_rf_time, vae_nb_time"

    write_fie(num_epoch, step, dataname, conf, header, VAE_loss_, VAE_recon_loss_, VAE_latent_loss_, VAE_auc_svm_,
              VAE_auc_dt_, VAE_auc_rf_, VAE_auc_nb_, VAE_t1_, VAE_t2_, VAE_t3_, VAE_t4_, 'VAE', algorithm)


# ---------------------------------------------------------------------------------------------
class DAE(object):
    def __init__(self, learning_rate=1e-3, batch_size=100, hidden_layers=[85, 30, 12], input_dim=0, noise_factor=0.25):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.noise_factor = noise_factor
        self.build(input_dim)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def build(self, input_dim):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, input_dim])
        Xnoise = self.x + self.noise_factor * tf.random_normal(tf.shape(self.x))
        Xnoise = tf.clip_by_value(Xnoise, 0., 1.)
        # Encode
        f1 = fc(Xnoise, self.hidden_layers[0], scope='dae_enc_fc1', activation_fn=tf.nn.relu)
        f2 = fc(f1, self.hidden_layers[1], scope='dae_enc_fc2', activation_fn=tf.nn.relu)
        self.z = fc(f2, self.hidden_layers[2], scope='dae_enc_fc3_mu', activation_fn=None)

        # Decode
        g1 = fc(self.z, self.hidden_layers[1], scope='dae_dec_fc2', activation_fn=tf.nn.relu)
        g2 = fc(g1, self.hidden_layers[0], scope='dae_dec_fc1', activation_fn=tf.nn.relu)

        self.x_hat = fc(g2, input_dim, scope='dae_dec_xhat', activation_fn=tf.nn.sigmoid)

        recon_loss = tf.reduce_mean(tf.square(self.x - self.x_hat), 1)
        self.recon_loss = tf.reduce_mean(recon_loss)
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.recon_loss)

        return

    def run_single_step(self, x):
        _, recon_loss = self.sess.run(
            [self.train_op, self.recon_loss],
            feed_dict={self.x: x}
        )
        return recon_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z


DAE_recon_loss_ = []
DAE_auc_svm_ = []
DAE_auc_dt_ = []
DAE_auc_rf_ = []
DAE_auc_nb_ = []
DAE_auc_kn_ = []
DAE_auc_logistic_ = []
DAE_t1_ = []
DAE_t2_ = []
DAE_t3_ = []
DAE_t4_ = []
DAE_t5_ = []
DAE_t6_ = []


def DAE_trainer(learning_rate=1e-3, batch_size=100, num_epoch=10, hidden_layers=[7, 4, 2],
                input_dim=0, step=20, X_train=[], X_test=[], Y_train=[], Y_test=[], dt=[], noise_factor=0.25):
    model1 = DAE(learning_rate=learning_rate, batch_size=batch_size, hidden_layers=hidden_layers, input_dim=input_dim,
                 noise_factor=noise_factor)
    for epoch in range(num_epoch):
        num_sample = len(X_train)
        for iter in range(num_sample // batch_size):
            X_mb, _ = dt.train.next_batch(batch_size)
            # Execute the forward and the backward pass and report computed losses
            recon_loss = model1.run_single_step(X_mb)

        if epoch % step == 0:
            chartcolumn.take_display_traning('Epoch ' + str(epoch) + ' Recon loss: ' + str(recon_loss))
            # model.writer.add_summary(summary, epoch )

            z_train = model1.transformer(X_train)
            s = time.time()
            z_test = model1.transformer(X_test)
            e = time.time()
            t_tr = (e - s) / float(len(X_test))

            auc_svm, t1 = classifier.svm(z_train, Y_train, z_test, Y_test)
            auc_dt, t2 = classifier.decisiontree(z_train, Y_train, z_test, Y_test)
            auc_rf, t3 = classifier.rf(z_train, Y_train, z_test, Y_test)
            auc_nb, t4 = classifier.naive_baves(z_train, Y_train, z_test, Y_test)
            auc_kn, t5 = classifier.KNeighbors(z_train, Y_train, z_test, Y_test)
            auc_logistic, t6 = classifier.Logistic(z_train, Y_train, z_test, Y_test)

            DAE_recon_loss_.append(recon_loss)

            DAE_auc_svm_.append(auc_svm)
            DAE_auc_dt_.append(auc_dt)
            DAE_auc_rf_.append(auc_rf)
            DAE_auc_nb_.append(auc_nb)
            DAE_auc_kn_.append(auc_kn)
            DAE_auc_logistic_.append(auc_logistic)
            
            DAE_t1_.append((t1 + t_tr))
            DAE_t2_.append((t2 + t_tr))
            DAE_t3_.append((t3 + t_tr))
            DAE_t4_.append((t4 + t_tr))
            DAE_t5_.append((t5 + t_tr))
            DAE_t6_.append((t6 + t_tr))

    print('Done DAE!')

    return model1


def call_DAE_train(data_index, input_dim, balance_rate, lr, block_size, num_epoch, hidden_layer,
                   step, X_train, X_test, Y_train, Y_test, dt, noise_factor, dataname, conf, algorithm, AUC_and_Structure):
    model = DAE_trainer(learning_rate=lr, batch_size=block_size, num_epoch=num_epoch,
                        hidden_layers=hidden_layer, input_dim=input_dim, step=step,
                        X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, dt=dt,
                        noise_factor=noise_factor)
    # save auc to file:
    save_auc_tofile(data_index, input_dim, balance_rate, "DAE", DAE_auc_svm_, DAE_auc_dt_, DAE_auc_rf_, DAE_auc_nb_,DAE_auc_kn_,DAE_auc_logistic_,DAE_t1_, DAE_t2_, DAE_t3_, DAE_t4_,DAE_t5_,DAE_t6_, dataname, AUC_and_Structure)
    header = "epoch, dae_loss, dae_svm, dae_dt, dae_rf, dae_nb, dae_kn, dae_logistic, dae_svm_time, dae_dt_time, dae_rf_time, dae_nb_time, dae_kn_time, dae_logistic_time"
    #write_fie(num_epoch, step, dataname, conf, header, [], DAE_recon_loss_, [], DAE_auc_svm_, DAE_auc_dt_, DAE_auc_rf_,DAE_auc_nb_,DAE_auc_kn_,DAE_auc_logistic_,DAE_t1_, DAE_t2_, DAE_t3_, DAE_t4_,DAE_t5_,DAE_t6_, "DAE", algorithm)


# ---------------------------------------------------------------------------------------------

def write_fie(num_epoch, step, dataname, conf, header, loss, recon_loss, latent_loss, auc_svm, auc_dt, auc_rf, auc_nb, auc_kn, auc_logistic, time_svm, time_dt, time_rf, time_nb, time_kn, time_logistic, method, algorithm):
    header = header
    t = np.arange(0, num_epoch, step)
    print(len(t))
    # CNN_auc_svm_ = [0.5] * len(AE_recon_loss_)
    # CNN_auc_dt_ = [0.5] * len(AE_recon_loss_)
    # CNN_auc_rf_ = [0.5] * len(AE_recon_loss_)

    if method == "AE" or method == 'DAE':
        np.savetxt('Results//RF_AUC_DIF/' + dataname + '/' + dataname + "_" + conf + "_" + method + ".csv",np.column_stack((t, recon_loss, auc_svm, auc_dt, auc_rf, auc_nb, auc_kn, auc_logistic, time_svm, time_dt, time_rf, time_nb, time_kn, time_logistic)),delimiter=",", fmt='%s', header=header)
        
        header = header.split(',')
        
        chartcolumn.chart_line_AUC_and_Time('Results//RF_AUC_DIF/' + dataname + '/' + dataname + "_" + conf + "_" + method + ".csv", header[2],header[3], header[4], header[5],dataname, method, algorithm, 'AUC')
        chartcolumn.chart_line_AUC_and_Time(
            'Results//RF_AUC_DIF/' + dataname + '/' + dataname + "_" + conf + "_" + method + ".csv", header[6],header[7], header[8], header[9],dataname, method, algorithm, 'Time')
       

