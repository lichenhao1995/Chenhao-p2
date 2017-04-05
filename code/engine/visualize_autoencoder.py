"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Visually compare autoencoder to naive compression scheme.
Usage: Run the command `python -m engine.visualize_autoencoder`
    Then enter in labels in [0, 7) into the prompt to visualize
    autoencoder behavior on a randomly selected image of a corresponding
    class. Specifically, shown side-by-side will be the original image,
    a naive reconstruction obtained by downsampling-then-upsampling, and
    the autoencoder reconstruction. Exit by causing a KeyboardInterrupt
    (press CTRL-c).
"""

import numpy as np
import tensorflow as tf
from model.build_autoencoder import naive, autoencoder
from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets
import matplotlib.pyplot as plt


def get_index_from_user_supplied_label(ys):
    ''' Return index (into validation set) corresponding to user-supplied
        label.
    '''
    while True:
        try:
            label = int(input('Enter label in [0, 7): '))
            assert(0 <= label < 7)
            break
        except ValueError:
            print('Oops! I need an integer...')
        except AssertionError:
            print('Oops! Valid labels are in [0, 7)...')
    while True:
        index = np.random.choice(len(ys))
        if ys[index][label]:
            return index


def plot(subplot_index, image, name, nb_subplots=3):
    ''' Plot a given image side-by-side the previously plotted ones. '''
    plt.subplot(1, nb_subplots, subplot_index + 1)
    plt.imshow(image, plt.get_cmap('gray'),
               interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title(name)
    plt.xticks([])
    plt.yticks([])


def user_interaction_loop(ys, orig_images, naive_recons, auto_recons):
    ''' Main loop: user enters labels to produce plots '''
    sl = get('MODEL.SQRT_REPR_DIM')
    try:
        while True:
            index = get_index_from_user_supplied_label(ys)
            plot(0, orig_images[index].reshape(32, 32), 'original image')
            plot(1, naive_recons[index].reshape(sl, sl), 'naive recon')
            plot(2, auto_recons[index].reshape(32, 32), 'autoencoder recon')
            plt.show()
    except KeyboardInterrupt:
        print('OK, bye!')

def test(test_org, test_rec):
    rec = tf.placeholder(tf.float32, shape=[None, 1024])  
    org = tf.placeholder(tf.float32, shape=[None, 1024])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(org, rec))))
    error = rmse.eval(feed_dict={org: test_org, rec: test_rec})  
    return error

def rmse(original,processed):
    num_image=original.shape[0]
    ret=np.zeros((num_image,))
    for idx in range(num_image):
        ret[idx]=np.sqrt(np.mean(np.square(original[idx]-processed[idx])))
    return ret

def grid_plot(subplot_index, original_image, processed_image, name, nb_subplots=3):
    plt.subplot(nb_subplots, 2, subplot_index*2-1)
    plt.imshow(original_image, plt.get_cmap('gray'),
               interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title(name+"_input")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(nb_subplots, 2, subplot_index*2)
    plt.imshow(processed_image, plt.get_cmap('gray'),
               interpolation='bicubic', clim=(-1.0, +1.0))
    plt.title(name+"_output")
    plt.xticks([])
    plt.yticks([])

# if __name__ == '__main__':
#     print('restoring model...')
#     assert is_file_prefix(
#         'TRAIN.AUTOENCODER.CHECKPOINT'), "training checkpoint not found!"
#     sess = tf.InteractiveSession()  # start talking to tensorflow backend
#     auto_orig, auto_repr, auto_recon = autoencoder()  # fetch autoencoder layers
#     naive_orig, naive_repr, naive_recon = naive()  # fetch naive baseline layers
#     saver = tf.train.Saver()  # prepare to restore weights
#     saver.restore(sess, get('TRAIN.AUTOENCODER.CHECKPOINT'))
#     print('Yay! I restored weights from a saved model!')

#     print('loading data...')
#     faces = read_data_sets()
#     ys = faces.validation.labels
#     Xs = faces.validation.images

#     print('computing reconstructions...')
#     Ns = naive_recon.eval(feed_dict={naive_orig: Xs})
#     As = auto_recon.eval(feed_dict={auto_orig: Xs})
    
#     overallError = test(Xs,As)
#     baselineError = test(Xs,np.zeros(Xs.shape))
#     print('***overall error', np.mean(overallError))
#     print('***Baseline error', np.mean(baselineError))
#     imSize = Xs.shape[0]
#     for label in range(7):
#         error = [overallError[idx] for idx in range(imSize) if ys[idx][label]]
#         index = [idx for idx in range(imSize) if ys[idx][label]]
#         print('****', label)
#         print('average', np.mean(error))
#         print('worst', np.max(error))
#         print('best', np.min(error))
#         print('typical', np.median(error))
#         typicalIndex = index[np.argsort(error)[len(error)//2]]
#         worstIndex = index[np.argmax(error)]
#         bestIndex = index[np.argmin(error)]
        
#         #plot the worst pair
#         plt.subplot(nb_subplots, 2, 1)
#         plt.imshow(Xs[worstIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("worst_case_input")
#         plt.subplot(nb_subplots, 2, 2)
#         plt.imshow(As[worstIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("worst_case_output")
#         #plot the best pair
#         plt.subplot(nb_subplots, 2, 3)
#         plt.imshow(Xs[bestIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("best_case_input")
#         plt.subplot(nb_subplots, 2, 4)
#         plt.imshow(As[bestIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("best_case_output")
#         #plot the typical pair
#         plt.subplot(nb_subplots, 2, 5)
#         plt.imshow(Xs[typicalIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("typical_case_input")
#         plt.subplot(nb_subplots, 2, 6)
#         plt.imshow(As[typicalIndex].reshape(32, 32), plt.get_cmap('gray'),
#                interpolation='bicubic', clim=(-1.0, +1.0))
#         plt.title("typical_case_output")
       
#     print('starting visualization...')
#     user_interaction_loop(ys, Xs, Ns, As)


if __name__ == '__main__':
    print('restoring model...')
    assert is_file_prefix(
        'TRAIN.AUTOENCODER.CHECKPOINT'), "training checkpoint not found!"
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    auto_orig, auto_repr, auto_recon = autoencoder()  # fetch autoencoder layers
    naive_orig, naive_repr, naive_recon = naive()  # fetch naive baseline layers
    saver = tf.train.Saver()  # prepare to restore weights
    saver.restore(sess, get('TRAIN.AUTOENCODER.CHECKPOINT'))
    print('Yay! I restored weights from a saved model!')

    print('loading data...')
    faces = read_data_sets()
    ys = faces.validation.labels
    Xs = faces.validation.images

    print('computing reconstructions...')
    Ns = naive_recon.eval(feed_dict={naive_orig: Xs})
    As = auto_recon.eval(feed_dict={auto_orig: Xs})

#Added Code
    baseline_rmse=rmse(Xs,np.zeros(Xs.shape))
    network_rmse=rmse(Xs,As)
    print('---Baseline test_error:{}'.format(np.mean(baseline_rmse)))
    print('---Network test_error:{}'.format(np.mean(network_rmse)))
    num_image=Xs.shape[0]
    for label in range(7):
        label_rmse=[network_rmse[idx] for idx in range(num_image) if ys[idx][label]]
        label_idx=[idx for idx in range(num_image) if ys[idx][label]]
        print('---label:{0}'.format(label))
        print('----avg:{0}\n----worst:{1}\n----best:{2}\n----typical:{3}'.format(
            np.mean(label_rmse),
            np.max(label_rmse),
            np.min(label_rmse),
            np.median(label_rmse)))
        worst_case_idx=label_idx[np.argmax(label_rmse)]
        best_case_idx=label_idx[np.argmin(label_rmse)]
        typical_case_idx=label_idx[np.argsort(label_rmse)[len(label_rmse)//2]]
        grid_plot(1,Xs[worst_case_idx].reshape(32, 32),As[worst_case_idx].reshape(32, 32),'worst_case')
        grid_plot(2,Xs[best_case_idx].reshape(32, 32),As[best_case_idx].reshape(32, 32),'best_case')
        grid_plot(3,Xs[typical_case_idx].reshape(32, 32),As[typical_case_idx].reshape(32, 32),'typical_case')
        plt.savefig("part3e_label{}.png".format(label))
        #plt.show()
    #print('starting interactive visualization...')
    #user_interaction_loop(ys, Xs, Ns, As)

