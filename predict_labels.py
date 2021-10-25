#!/usr/bin/env python3

import sys
import math
import numpy as np
from os.path import isfile, join as join_path
from os import makedirs
import csv

from scipy.sparse import lil_matrix

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.config import experimental as tfconf_exp
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import F1Score

from transformers import AutoConfig, AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification, AdamWeightDecay
from transformers.optimization_tf import create_optimizer

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from logging import warning

from readers import READERS, get_reader
from common import timed


# Parameter defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEQ_LEN = 512
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_PROPORTION = 0.1 


def init_tf_memory():
    gpus = tfconf_exp.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tfconf_exp.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument('--model_name', default=None,
                    help='pretrained model name')
    ap.add_argument('--train', metavar='FILE', required=False,
                    help='training data')
    ap.add_argument('--dev', metavar='FILE', required=False,
                    help='development data')
    ap.add_argument('--test', metavar='FILE', required=False,
                    help='test data', default=None)
    ap.add_argument('--bg_train', metavar='FILE', required=False,
                    help='background corpus for training', default=None)
    ap.add_argument('--bg_sample_rate', metavar='FLOAT', type=float,
                    default=1.,
                    help='rate at which to sample from background corpus (X:1)')
    ap.add_argument('--batch_size', metavar='INT', type=int,
                    default=DEFAULT_BATCH_SIZE,
                    help='batch size for training')
    ap.add_argument('--epochs', metavar='INT', type=int, default=1,
                    help='number of training epochs')
    ap.add_argument('--epoch_len', metavar='INT', type=int, default=None,
                    help='override epoch length with data generator')
    ap.add_argument('--lr', '--learning_rate', metavar='FLOAT', type=float,
                    default=DEFAULT_LR, help='learning rate')
    ap.add_argument('--seq_len', metavar='INT', type=int,
                    default=DEFAULT_SEQ_LEN,
                    help='maximum input sequence length')
    ap.add_argument('--warmup_proportion', metavar='FLOAT', type=float,
                    default=DEFAULT_WARMUP_PROPORTION,
                    help='warmup proportion of training steps')
    ap.add_argument('--input_format', choices=READERS.keys(),
                    default=list(READERS.keys())[0],
                    help='input file format')
    ap.add_argument('--multiclass', default=False, action='store_true',
                    help='task has exactly one label per text')
    ap.add_argument('--save_model', default=None, metavar='FILE',
                    help='save model to file')
    ap.add_argument('--save_weights', default=None, metavar='FILE',
                    help='save model weights to file')
    ap.add_argument('--save_predictions', default=None, metavar='FILE',
                    help='save predictions and labels for dev set, or for test set if provided')
    #ap.add_argument('--save_predictions', default=False, action='store_true',
    #                help='save predictions and labels for dev set, or for test set if provided')
    ap.add_argument('--load_model', default=None, metavar='FILE',
                    help='load model from file')
    ap.add_argument('--load_weights', default=None, metavar='FILE',
                    help='load model weights from file')
    ap.add_argument('--load_labels', default=None, metavar='FILE',
                    help='load labels from file')
    ap.add_argument('--threshold', metavar='FLOAT', type=float, default=None,
                    help='fixed threshold for multilabel prediction')
    ap.add_argument('--log_file', default="train.log", metavar='FILE',
                    help='log parameters and performance to file')
    ap.add_argument('--test_log_file', default="test.log", metavar='FILE',
                    help='log parameters and performance on test set to file')
    return ap



def load_pretrained(options):
    name = options.model_name
    config = AutoConfig.from_pretrained(name)
    config.return_dict = True
    tokenizer = AutoTokenizer.from_pretrained(name, config=config)
    model = TFAutoModel.from_pretrained(name, config=config)
    #model = TFAutoModelForSequenceClassification.from_pretrained(name, config=config)

    if options.seq_len > config.max_position_embeddings:
        warning(f'--seq_len ({options.seq_len}) > max_position_embeddings '
                f'({config.max_position_embeddings}), using latter')
        options.seq_len = config.max_position_embeddings

    return model, tokenizer, config


def get_custom_objects():
    """Return dictionary of custom objects required for load_model."""
    from transformers import AdamWeightDecay
    from tensorflow_addons.metrics import F1Score
    return {
        'AdamWeightDecay': AdamWeightDecay,
        'F1Score': F1Score,
    }


def load_trained_model(directory):
    config = AutoConfig.from_pretrained(directory)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        config=config
    )
    model = load_model(
        join_path(directory, 'model.hdf5'),
        custom_objects=get_custom_objects()
    )
    labels = []
    with open(join_path(directory, 'labels.txt')) as f:
        for ln, l in enumerate(f, start=1):
            labels.append(l.rstrip('\n'))
    return model, tokenizer, labels, config


def get_pretrained_model_main_layer(model):
    # Transformers doesn't support saving models wrapped in keras
    # (https://github.com/huggingface/transformers/issues/2733) at the
    # time of this writing. As a workaround, use the main layer
    # instead of the model. As the main layer has different names for
    # different models (TFBertModel.bert, TFRobertaModel.roberta,
    # etc.), this has to check which model we're dealing with.
    from transformers import TFBertModel, TFRobertaModel

    if isinstance(model, TFBertModel):
        return model.bert
    elif isinstance(model, TFRobertaModel):
        return model.roberta
    else:
        raise NotImplementedError(f'{model.__class__.__name__}')


@timed
def save_trained_model(directory, model, tokenizer, labels, config):
    makedirs(directory, exist_ok=True)
    model.save(join_path(directory, 'model.hdf5'))
    config.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    with open(join_path(directory, 'labels.txt'), 'w') as out:
        for label in labels:
            print(label, file=out)


def get_optimizer(num_train_examples, options):
    steps_per_epoch = math.ceil(num_train_examples / options.batch_size)
    num_train_steps = steps_per_epoch * options.epochs
    num_warmup_steps = math.floor(num_train_steps * options.warmup_proportion)

    # Mostly defaults from transformers.optimization_tf
    optimizer, lr_scheduler = create_optimizer(
        options.lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        min_lr_ratio=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay_rate=0.01,
        power=1.0,
    )
    return optimizer


def build_classifier(pretrained_model, num_labels, optimizer, options):
    # workaround for lack of support for saving pretrained models
    pretrained_model = get_pretrained_model_main_layer(pretrained_model)

    seq_len = options.seq_len
    input_ids = Input(
        shape=(seq_len,), dtype='int32', name='input_ids')
    attention_mask = Input(
        shape=(seq_len,), dtype='int32', name='attention_mask')
    inputs = [input_ids, attention_mask]

    pretrained_outputs = pretrained_model(inputs)
    pooled_output = pretrained_outputs['last_hidden_state'][:,0,:] #CLS

    # TODO consider Dropout here
    if options.multiclass:
        output = Dense(num_labels, activation='softmax')(pooled_output)
        loss = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy(name='acc')]
    else:
        output = Dense(num_labels, activation='sigmoid')(pooled_output)
        loss = BinaryCrossentropy()
        metrics = [
            F1Score(name='f1_th0.5', num_classes=num_labels, average='micro', threshold=0.5)
        ]

    model = Model(
        inputs=inputs,
        outputs=[output]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


@timed
def load_data(fn, options, max_chars=None):
    read = get_reader(options.input_format)
    texts, labels = [], []
    with open(fn) as f:
        for ln, (text, text_labels) in enumerate(read(f, fn), start=1):
            if options.multiclass and not text_labels:
                raise ValueError(f'missing label on line {ln} in {fn}: {l}')
            elif options.multiclass and len(text_labels) > 1:
                raise ValueError(f'multiple labels on line {ln} in {fn}: {l}')
            texts.append(text[:max_chars])
            labels.append(text_labels)
    print(f'loaded {len(texts)} examples from {fn}', file=sys.stderr)
    return texts, labels


class DataGenerator(Sequence):
    """ Data generator for sampling from a main and any number of background corpora, e.g. uniformly from multiple languages. """
    def __init__(self, data_path, tokenize_func, options, bg_data_path=None, bg_sample_rate=1., max_chars=None, label_encoder=None, epoch_len=None):
        ## Load data
        texts, labels = load_data(data_path, options, max_chars=max_chars)
        self.num_examples = len(texts)
        self.batch_size = options.batch_size
        self.epoch_len = epoch_len
        self.X = tokenize_func(texts)
        assert self.X['attention_mask'].min() >= 0
        assert self.X['attention_mask'].max() <= 1
        all_labels = labels

        if bg_data_path is not None:
            self.bg_sample_rate = bg_sample_rate
            self.bg_num_examples, self.bg_X, self.bg_Y = [], [], []
            for path in bg_data_path.split():
                bg_texts, bg_labels = load_data(path, options, max_chars=max_chars)
                self.bg_num_examples.append(len(bg_texts))
                self.bg_X.append(tokenize_func(bg_texts))
                for bg_x in self.bg_X:
                    assert bg_x['attention_mask'].min() >= 0
                    assert bg_x['attention_mask'].max() <= 1
                all_labels += bg_labels
                self.bg_Y.append(bg_labels)

        ## Init label encoder
        if label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            if options.load_labels is None:
                self.label_encoder.fit(all_labels)
            else:
                label_set = [[l.strip()] for l in open(options.load_labels).readlines() if l]
                self.label_encoder.fit(label_set)
            #print("Classes:", self.label_encoder.classes_)
            print("Number of classes:", len(self.label_encoder.classes_))
        else:
            self.label_encoder = label_encoder

        self.Y = self.label_encoder.transform(labels)
        self.num_labels = len(self.label_encoder.classes_)
        assert self.Y.max() < self.num_labels
        assert self.Y.min() >= 0

        if bg_data_path is not None:
            self.bg_Y = [self.label_encoder.transform(y) for y in self.bg_Y]
            for bg_y in self.bg_Y:
                assert bg_y.max() < self.num_labels
                assert bg_y.min() >= 0


            #self.bg_num_labels = len(self.label_encoder.classes_)
            self.bg_num_corpora = len(self.bg_num_examples)
        else:
            self.bg_sample_rate = 0
            self.bg_num_examples = [0]
            self.bg_num_corpora = 0

        self.on_epoch_end()


    def on_epoch_end(self):
        self.indexes = np.arange(self.num_examples)
        np.random.shuffle(self.indexes)

        if self.bg_num_corpora > 0:# and self.bg_sample_rate > 0:
            if hasattr(self, "bg_indexes"):
                for i, bg_indexes in enumerate(self.bg_indexes):
                    seen_bg_idxs = bg_indexes[:len(self.indexes)]
                    unseen_bg_idxs = bg_indexes[len(self.indexes):]
                    np.random.shuffle(seen_bg_idxs)
                    self.bg_indexes[i] = np.concatenate([unseen_bg_idxs, seen_bg_idxs])
            else:
                self.bg_indexes = [np.arange(x) for x in self.bg_num_examples]
                for i,_ in enumerate(self.bg_indexes):
                    np.random.shuffle(self.bg_indexes[i])

        self.index = 0

    def __len__(self):
        if self.epoch_len is not None:
            return self.epoch_len
        else:
            #return int((self.num_examples//self.batch_size)*(1+self.bg_sample_rate)) * self.bg_num_corpora
            return int((self.num_examples//self.batch_size) * (self.bg_sample_rate*(1+self.bg_num_corpora)))

    def __getitem__(self, index):
        try:
            limit = 1/(self.bg_sample_rate+self.bg_num_corpora)
        except ZeroDivisionError:
            limit = 1.0
        if np.random.random() <= limit or self.bg_num_corpora == 0:
            batch_indexes = self.indexes[self.index*self.batch_size:(self.index+1)*self.batch_size]
            if len(batch_indexes) < self.batch_size:
                batch_indexes = np.concatenate([batch_indexes, self.indexes[:self.batch_size-len(batch_indexes)]])
                end = ((self.index+1)*self.batch_size) % len(self.indexes)
                if end < self.batch_size:
                    end = self.batch_size
                    beg = 0
                else:
                    beg = end-self.batch_size
                batch_indexes = self.indexes[beg:end]
            self.index += 1
            X, Y = self.X, self.Y
        else:
            i = np.random.randint(0, self.bg_num_corpora)
            batch_indexes = self.bg_indexes[i][self.index*self.batch_size:(self.index+1)*self.batch_size]

            if len(batch_indexes) < self.batch_size:
                end = ((self.index+1)*self.batch_size) % len(self.bg_indexes[i])
                if end < self.batch_size:
                    end = self.batch_size
                    beg = 0
                else:
                    beg = end-self.batch_size
                batch_indexes = self.bg_indexes[i][beg:end]

            X, Y = self.bg_X[i], self.bg_Y[i]

        batch_X = {}
        for key in self.X:
            batch_X[key] = np.empty((self.batch_size, *X[key].shape[1:]))
            for j, idx in enumerate(batch_indexes):
                batch_X[key][j] = X[key][idx]

        batch_y = np.empty((self.batch_size, *Y.shape[1:]), dtype=int)
        for j, idx in enumerate(batch_indexes):
            batch_y[j] = Y[idx]

        assert batch_y.max() < self.num_labels
        assert batch_y.min() >= 0
        return batch_X, batch_y


def make_tokenization_function(tokenizer, options):
    seq_len = options.seq_len
    @timed
    def tokenize(text):
        tokenized = tokenizer(
            text,
            max_length=seq_len,
            truncation=True,
            padding=True,
            return_tensors='np'
        )
        # Return dict b/c Keras (2.3.0-tf) DataAdapter doesn't apply
        # dict mapping to transformer.BatchEncoding inputs
        return {
            'input_ids': tokenized['input_ids'],
#            'token_type_ids': tokenized['token_type_ids'],
            'attention_mask': tokenized['attention_mask'],
        }
    return tokenize


@timed
def prepare_classifier(num_train_examples, num_labels, options):
    pretrained_model, tokenizer, config = load_pretrained(options)
    optimizer = get_optimizer(num_train_examples, options)
    model = build_classifier(pretrained_model, num_labels, optimizer, options)
    return model, tokenizer, optimizer


def optimize_threshold(model, train_X, train_Y, test_X, test_Y, options=None, epoch=None, save_pred_to=None, return_auc=False):
    print("Predicting on train set...")
    labels_prob = model.predict(train_X, verbose=1)#, batch_size=options.batch_size)

    best_f1 = 0.
    print("Optimizing threshold...\nThres.\tPrec.\tRecall\tF1")
    # Test 0.5, 0.55, 0.45...
    for threshold in [0.5+((i+1)//2)/20.*(i%2*2-1) for i in range(9)]: #np.arange(0.3, 0.7, 0.05):
        labels_pred = lil_matrix(labels_prob.shape, dtype='b')
        labels_pred[labels_prob>=threshold] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(train_Y, labels_pred, average="micro")
        print("%.2f\t%.4f\t%.4f\t%.4f" % (threshold, precision, recall, f1), end="")
        if f1 > best_f1:
            print("\t*")
            best_f1 = f1
            #best_f1_epoch = epoch
            best_f1_threshold = threshold
        else:
            print()

    #print("Current F_max:", best_f1, "epoch", best_f1_epoch+1, "threshold", best_f1_threshold, '\n')
    #print("Current F_max:", best_f1, "threshold", best_f1_threshold, '\n')

    print("Predicting on evaluation set...")
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=best_f1_threshold] = 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (best_f1_threshold, epoch_str, test_precision, test_recall, test_f1))

    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy" % epoch)
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    if return_auc:
        auc = roc_auc_score(test_Y, test_labels_prob, average = 'micro')
        return test_f1, best_f1_threshold, test_labels_pred, auc
    else:
        return test_f1, best_f1_threshold, test_labels_pred


def test_threshold(model, test_X, threshold=0.4, options=None, epoch=None, return_auc=False, save_pred_to=None):
    test_labels_prob = model.predict(test_X, verbose=1)#, batch_size=options.batch_size)
    test_labels_pred = lil_matrix(test_labels_prob.shape, dtype='b')
    test_labels_pred[test_labels_prob>=threshold] = 1
    #test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_Y, test_labels_pred, average="micro")
    if epoch:
        epoch_str = ", epoch %d" % epoch
    else:
        epoch_str = ""
    #print("\nValidation/Test performance at threshold %.2f%s: Prec. %.4f, Recall %.4f, F1 %.4f" % (threshold, epoch_str, test_precision, test_recall, test_f1))

    if save_pred_to is not None:
        if epoch is None:
            print("Saving predictions to", save_pred_to+".*.npy")
            np.save(save_pred_to+".preds.npy", test_labels_pred.toarray())
            #np.save(save_pred_to+".gold.npy", test_Y)
            #np.save(save_pred_to+".class_labels.npy", label_encoder.classes_)
        else:
            print("Saving predictions to", save_pred_to+"-epoch%d.*.npy" % epoch)
            np.save(save_pred_to+"-epoch%d.preds.npy" % epoch, test_labels_pred.toarray())
            #np.save(save_pred_to+"-epoch%d.gold.npy" % epoch, test_Y)
            #np.save(save_pred_to+"-epoch%d.class_labels.npy" % epoch, label_encoder.classes_)

    
    return test_labels_pred


def test_auc(model, test_X, test_Y):
    labels_prob = model.predict(test_X, verbose=1)
    return roc_auc_score(test_Y, labels_prob, average = 'micro')


class Logger:
    def __init__(self, filename, model, params):
        self.filename = filename
        self.model = model
        self.log = dict([('p%s'%p, v) for p, v in params.items()])

    def record(self, epoch, logs):
        for k in logs:
            self.log['_%s' % k] = logs[k]
        self.log['_Epoch'] = epoch
        self.write()

    def write(self):
        file_exists = isfile(self.filename)
        with open(self.filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, sorted(self.log.keys()))
            if not file_exists:
                print("Creating log file", self.filename, flush=True)
                writer.writeheader()
            writer.writerow(self.log)


class EvalCallback(Callback):
    def __init__(self, model, train, dev, test=None, logfile="train.log", test_logfile=None, save_pred_to=None, params={}, patience=None, save_model_to=None, th_opt_train_batches=200, min_improvement=0.0, threshold=None):
        self.model = model
        #self.train_X = train_X
        #self.train_Y = train_Y
        self.train_X = {}
        self.train_Y = None
        for i, elem in enumerate(train): # Generate train sub set for optimizing threshold
            if i >= th_opt_train_batches:
                break
            x, y = elem
            for k in x:
                if k not in self.train_X:
                    self.train_X[k] = x[k]
                else:
                    self.train_X[k] = np.append(self.train_X[k], x[k], axis=0)
            if self.train_Y is None:
                self.train_Y = y
            else:
                self.train_Y = np.append(self.train_Y, y, axis=0)

        self.dev_X, self.dev_Y = dev
        if test is None:
            self.test_X, self.test_Y = None, None
        else:
            if type(test) is list or type(test) is tuple:
                self.test_X, self.test_Y = test
            else:
                self.test_X, self.test_Y = [test[0]], [test[1]]

        self.logger = Logger(logfile, self.model, params)
        if test_logfile is not None:
            print("Setting up test set logging to", test_logfile, flush=True)
            self.test_logger = Logger(test_logfile, self.model, params)
        if save_pred_to is not None:
            self.save_pred_to = save_pred_to
        else:
            self.save_pred_to = None
        if save_model_to is not None:
            self.save_model_to = save_model_to
        else:
            self.save_model_to = None
        self.best_f1 = -1
        self.patience = patience
        self.patience_left = patience
        self.min_improvement = min_improvement
        self.threshold = threshold
        self.model.best_f1 = -1
        self.model.best_auc = -1

    def on_epoch_end(self, epoch, logs={}):
        print("\nPerformance on validation set:")

        if self.threshold is None:
            logs['f1'], dev_th, _, logs['rocauc'] = optimize_threshold(self.model, self.train_X, self.train_Y, self.dev_X, self.dev_Y, epoch=epoch, return_auc=True)
        else:
            logs['f1'], _, _, logs['rocauc'] = test_threshold(self.model, self.dev_X, self.dev_Y, epoch=epoch, return_auc=True, threshold=self.threshold)

        #logs['rocauc'] = test_auc(self.model, self.dev_X, self.dev_Y)
        print("AUC dev:", logs['rocauc'])

        if self.test_X is not None:
            for test_i, (test_X, test_Y) in enumerate(zip(self.test_X, self.test_Y)):
                print("\nPerformance on test set %d:" % test_i)
                try:
                    if self.threshold is None:
                        #test_f1, _, _, test_auc_score = optimize_threshold(self.model, self.train_X, self.train_Y, test_X, test_Y, epoch=epoch, save_pred_to=self.save_pred_to, return_auc=True)
                        test_f1, test_th, test_pred, test_auc_score = test_threshold(self.model, test_X, test_Y, threshold=dev_th, return_auc=True)
                        print("Test F1:", test_f1)
                    else:
                        test_f1, _, _, test_auc_score = test_threshold(self.model, self.dev_X, self.dev_Y, epoch=epoch, return_auc=True, threhold=self.threshold)
                    #test_auc_score = test_auc(self.model, self.test_X, self.test_Y)
                    print("AUC test:", test_auc_score)
                except:
                    print("Evaluation on test set failed.")
                    test_f1, test_auc_score = np.nan, np.nan
        else:
            test_f1, test_auc_score = None, None

        if logs['f1'] > self.best_f1 + self.min_improvement:
            print("F1 score improved.")
            self.best_f1 = logs['f1']
            self.model.best_f1 = logs['f1']
            self.model.best_auc = logs['rocauc']
            self.best_score_epoch = epoch
            self.best_score_logs = logs
            self.best_test_f1 = test_f1
            self.best_test_auc = test_auc_score

            if self.save_model_to is not None:
                try:
                    print("Saving weights to", self.save_model_to)
                    self.model.save_weights(self.save_model_to)
                    #print("Saving model to", self.save_model_to+'.model')
                    #self.model.save(options.load_model+'.model')
                except:
                    pass
            if self.patience is not None:
                self.patience_left = self.patience
        else:
            print("F1 score not improved.")
            if self.patience is not None:
                if self.patience_left <= 0:
                    print("Patience elapsed. Best score: %.4f" % self.best_f1)
                    print("Stopping training...")
                    self.model.stop_training = True
                else:
                    print("Patience left:", self.patience_left)
                    self.patience_left -= 1
        print()

        if self.model.stop_training:
            epoch = self.best_score_epoch
            logs = self.best_score_logs
            test_f1 = self.best_test_f1
            test_auc_score = self.best_test_auc
            self.logger.record(epoch, logs)
            if self.test_X is not None:
                self.test_logger.record(epoch, {'f1': test_f1, 'rocauc': test_auc_score})

            self.flush_log_after_max_epochs = False
        else:
            self.test_logs = {'f1': test_f1, 'rocauc': test_auc_score}
            self.flush_log_after_max_epochs = True

        self.final_epoch = epoch

        #self.logger.record(epoch, logs)

    def on_train_end(self, logs):
        if self.flush_log_after_max_epochs:
            self.logger.record(self.final_epoch, self.best_score_logs)
            if self.test_X is not None:
                self.test_logger.record(self.final_epoch, self.test_logs)


def main(argv):
    init_tf_memory()
    options = argparser().parse_args(argv[1:])

    ### Load data without generator (needed for threshold optimization)
    if options.train is not None:
        train_texts, train_labels = load_data(options.train, options, max_chars=25000)
    if options.dev is not None:
        dev_texts, dev_labels = load_data(options.dev, options, max_chars=25000)
    
    if options.test is not None:
        test_data = [load_data(path, options) for path in options.test.split(';')]
        print("Loaded test data")

    pretrained_model, tokenizer, model_config = load_pretrained(options)
    tokenize = make_tokenization_function(tokenizer, options)

    if options.train is not None:
        train_X = tokenize(train_texts)
    if options.dev is not None:
        dev_X = tokenize(dev_texts)

    if options.test is not None:
        test_Xs = [tokenize(test_texts) for test_texts, _ in test_data]
        print("Tokenised test data")
    
    if options.bg_train is None:
        train_gen = DataGenerator(options.test, tokenize, options, max_chars=25000)
    else:
        train_gen = DataGenerator('data/eacl/en/train.tsv', tokenize, options, max_chars=25000, bg_data_path=options.bg_train, bg_sample_rate=options.bg_sample_rate, epoch_len=options.epoch_len)

    optimizer = get_optimizer(train_gen.num_examples*(1+train_gen.bg_sample_rate*train_gen.bg_num_corpora), options)
    classifier = build_classifier(pretrained_model, train_gen.num_labels, optimizer, options)

    if options.train is not None:
        train_Y = train_gen.label_encoder.transform(train_labels)
    if options.dev is not None:
        dev_Y = train_gen.label_encoder.transform(dev_labels)

    if options.load_weights is not None:
        print("Loading weights from", options.load_weights)
        classifier.load_weights(options.load_weights)
    elif options.load_model is not None:
        print("Loading model from", options.load_model)
        classifier, tokenizer, labels, config = load_trained_model(options.load_model)

    if options.train is None:
        assert (options.load_weights is not None or options.load_model is not None) and options.train is None
        print("No train, evaluating")# Evaluate only when no train data is passed

        if options.dev is not None:
            print("Evaluating on dev set...")
            if options.threshold is None:
                f1, th, dev_pred = optimize_threshold(classifier, train_X, train_Y, dev_X, dev_Y, options)
            else:
                f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=options.threshold)
            print("AUC dev:", test_auc(classifier, dev_X, dev_Y))

        if options.test is not None:
            print("Evaluating on test set...")
            for i in range(len(test_Xs)):
                print("  %d" % i)
                test_X = test_Xs[i]
                test_pred = test_threshold(classifier, test_X, threshold=options.threshold, save_pred_to=options.save_predictions)

            np.save(options.save_predictions+".class_labels.npy", train_gen.label_encoder.classes_) ## TODO: move? train_gen dep?
            #test_f1, test_th, test_pred = optimize_threshold(classifier, train_X, train_Y, test_X, test_Y, options)
            #print("AUC test:", test_auc(classifier, test_X, test_Y))

        #f1, th, dev_pred = test_threshold(classifier, dev_X, dev_Y, threshold=0.4)
        return


    callbacks = [] #[ModelCheckpoint(options.save_weights+'.{epoch:02d}', save_weights_only=True)]
    if options.threshold is None:
        if options.test is not None and options.test_log_file is not None:
            print("Initializing evaluation with dev and test set...")
            callbacks.append(EvalCallback(classifier, train_gen, (dev_X, dev_Y), test=(test_Xs, test_Ys),
                                        patience=5,
                                        threshold=options.threshold,
                                        logfile=options.log_file,
                                        test_logfile=options.test_log_file,
                                        save_pred_to=options.save_predictions,
                                        save_model_to=options.save_weights,
                                        params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
        else:
            print("Initializing evaluation with dev set...")
            callbacks.append(EvalCallback(classifier, train_gen, (dev_X, dev_Y),
                                        patience=5,
                                        threshold=options.threshold,
                                        logfile=options.log_file,
                                        save_model_to=options.save_weights,
                                        params={'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size}))
    else:
        print("Initializing early stopping criterion...")
        callbacks.append(EarlyStopping(monitor="val_f1_th0.5", verbose=1, patience=5, mode="max", restore_best_weights=True))


    """try:
        print("Best F1 during training:", max(history.history['val_f1_th0.5']))
    except:
        pass"""
    #print("Best AUC during training:", classifier.best_auc)
    #classifier = load_model(options.load_model+'.model')

    if options.threshold is not None:
        if options.dev is not None:
            f1, _, _, rocauc = test_threshold(classifier, dev_X, dev_Y, return_auc=True, threshold=options.threshold)
            print("Restored best checkpoint, F1: %.6f, AUC: %.6f" % (f1, rocauc))

            logger = Logger(options.log_file, None, {'LR': options.lr, 'N_epochs': options.epochs, 'BS': options.batch_size})
            try:
                epoch = len(history.history['loss'])-1
            except:
                epoch = -1
            logger.record(epoch, {'f1': f1, 'rocauc': rocauc})

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
