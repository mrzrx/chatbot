import sys
import time
import random

import tensorflow as tf
import numpy as np
import jieba

import word_token


PATH_CORPUS = '../corpus/corpus_xhj_10000.txt'  # 语料路径
MODEL_PATH = './model/'  #模型保存路径
MODEL_NAME = 'model'  #模型名字
PAD_ID = 0  # 空值填充0
GO_ID = 1  # 输出序列起始标记
EOS_ID = 2  # 结尾标记
UNK_ID = 3  # 未登录词
MIN_FREQUENCY = 10  # 词频大于等于该数的词才会有单独ID，其它统一为UNK

EMBEDDING_SIZE = 128  # embedding维度
NUM_UNITS = 128  # lstm隐藏层单元数
NUM_LAYERS = 2  # 隐藏层（lstm）层数
BEAM_WIDTH = 2  # beam搜索的宽度

BATCH_SIZE = 100  # batch大小
MAX_GRADIENT = 5  # 用于梯度修剪
LEARNING_RATE = 0.0003  # 学习率
LEARNING_DECAY_STEPS = 100  # 每隔多少步学习率下降
LEARNING_DECAY_RATE = 0.99  #学习率下降率
TRAIN_STEPS = 100000  # 一共训练多少步
PRINT_STEPS = 100  # 每隔多少步打印训练信息

wordToken = word_token.WordToken()
max_token_id = wordToken.load_file_list(PATH_CORPUS, MIN_FREQUENCY)
vocab_size = max_token_id + 4 + 1  # pad go eos unk 4个，索引由0开始所以加一 


def get_id_list(sentence):
    sentence_id_list = []
    word_list = jieba.cut(sentence)
    for word in word_list:
        sentence_id_list.append(wordToken.word2id(word))
    return sentence_id_list


def get_train_set():
    '''
    train_set = [[q1,a1], [q2,a2], ...]
    '''
    train_set = []
    with open(PATH_CORPUS, 'r', encoding='UTF-8') as corpus_file:
        while True:
            question = corpus_file.readline()
            answer = corpus_file.readline()
            if question and answer:
                question = question.strip()
                answer = answer.strip()
                question_id_list = get_id_list(question)
                answer_id_list = get_id_list(answer)
                if len(question_id_list) > 0 and len(answer_id_list) > 0:
                    train_set.append([question_id_list, answer_id_list])
            else:
                break
    return train_set


def get_batch(train_set, BATCH_SIZE):
    encoder_inputs = []
    decoder_inputs = []
    decoder_outputs = []
    target_weights = []
    if BATCH_SIZE >= len(train_set):
        batch_train_set = train_set
    else:
        batch_train_set = []
        for i in range(BATCH_SIZE):
            random_id = random.randint(0, len(train_set) - 1)
            batch_train_set.append(train_set[random_id])

    encoder_inputs_length = [len(n[0]) for n in batch_train_set]
    decoder_inputs_length = [len(n[1])+1 for n in batch_train_set]
    input_seq_len = max(encoder_inputs_length)
    output_seq_len = max(decoder_inputs_length)

    for sample in batch_train_set:
        encoder_inputs.append(sample[0] + [PAD_ID] * (input_seq_len - len(sample[0])))
        decoder_inputs.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))
        decoder_outputs.append(sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1) + [EOS_ID])
    for decoder_output in decoder_outputs:
        target_weights.append([0.0 if decoder_output[length_idx] == PAD_ID else 1.0 for length_idx in range(output_seq_len)])
    return encoder_inputs, encoder_inputs_length, decoder_inputs, decoder_inputs_length, decoder_outputs, target_weights


def model(encoder_inputs, encoder_lengths, vocab_size, decoder_inputs=None, decoder_lengths=None, mode='inference'):
    # Embedding
    embedding = tf.get_variable("embedding", [vocab_size, EMBEDDING_SIZE])
    encoder_emb_inp = tf.nn.embedding_lookup(embedding, encoder_inputs)
    if mode=='train':
        decoder_emb_inp = tf.nn.embedding_lookup(embedding, decoder_inputs)

    # encoder
    forward_encoder_cell = [tf.nn.rnn_cell.LSTMCell(n) for n in [NUM_UNITS]*NUM_LAYERS]
    forward_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(forward_encoder_cell)
    backward_encoder_cell = [tf.nn.rnn_cell.LSTMCell(n) for n in [NUM_UNITS]*NUM_LAYERS]
    backward_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(backward_encoder_cell)
    bi_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(forward_encoder_cell, backward_encoder_cell, encoder_emb_inp, 
                                                                    sequence_length=encoder_lengths, dtype=tf.float32)
    encoder_outputs = tf.concat(bi_outputs, -1)
    encoder_state = []
    for i in range(NUM_LAYERS):
        encoder_state.append(bi_encoder_state[0][i]) 
        encoder_state.append(bi_encoder_state[1][i])  
    encoder_state = tuple(encoder_state)
    encoder_state = encoder_state[-NUM_LAYERS:]
    
    # decoder
    if mode=='train':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(NUM_UNITS, encoder_outputs, memory_sequence_length=encoder_lengths)
        decoder_cell = [tf.nn.rnn_cell.LSTMCell(n) for n in [NUM_UNITS]*NUM_LAYERS]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=NUM_UNITS)
        decoder_initial_state = decoder_cell.zero_state(BATCH_SIZE, dtype=tf.float32).clone(cell_state=encoder_state)
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=tf.layers.Dense(vocab_size, use_bias=False))
    else:
        encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=BEAM_WIDTH)
        encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=BEAM_WIDTH)
        encoder_lengths = tf.contrib.seq2seq.tile_batch(self.encoder_lengths, multiplier=BEAM_WIDTH)
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(NUM_UNITS, encoder_outputs, memory_sequence_length=encoder_lengths)
        decoder_cell = [tf.nn.rnn_cell.LSTMCell(n) for n in [NUM_UNITS]*NUM_LAYERS]
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=NUM_UNITS)
        decoder_initial_state = decoder_cell.zero_state(BATCH_SIZE*BEAM_WIDTH, dtype=tf.float32).clone(cell_state=encoder_state)
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                       embedding=embedding,
                                                       start_tokens=GO_ID,
                                                       end_token=EOS_ID,
                                                       initial_state=decoder_initial_state,
                                                       beam_width=BEAM_WIDTH,
                                                       output_layer=tf.layers.Dense(vocab_size, use_bias=False),
                                                       length_penalty_weight=0.0,
                                                       coverage_penalty_weight=0.0)
    outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations = tf.reduce_max(decoder_lengths) if mode=='train' else 2*tf.reduce_max(encoder_lengths))
    if mode=='train':
        logits = outputs.rnn_output
    else:
        answers = outputs.predicted_ids[:,:,0]
    return logits if mode=='train' else answers


def train():
    global LEARNING_RATE
    encoder_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='encoder_inputs')
    encoder_lengths = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='encoder_inputs_length')
    decoder_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='decoder_inputs')
    decoder_lengths = tf.placeholder(tf.int32, shape=(BATCH_SIZE,), name='decoder_inputs_length')
    decoder_outputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE, None], name='decoder_outputs')
    target_weights = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None], name='target_weights')

    logits = model(encoder_inputs, encoder_lengths, vocab_size, decoder_inputs, decoder_lengths, 'train')
    # loss and optimization
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
    train_loss = (tf.reduce_sum(crossent * target_weights) / BATCH_SIZE)
    # train_loss = (tf.reduce_sum(crossent * target_weights) / (batch_size*num_time_steps))  # plays down the errors made on short sentences
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRADIENT)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
    saver = tf.train.Saver()

    train_set = get_train_set()
    with tf.Session() as sess:
        pre_step = 0
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
            pre_step = int(ckpt.model_checkpoint_path.split('-')[1])
        else:
            sess.run(tf.global_variables_initializer())
            
        for step in range(TRAIN_STEPS):
            step += pre_step
            batch_encoder_inputs, batch_encoder_lengths, batch_decoder_inputs, batch_decoder_lengths, batch_decoder_outputs, batch_target_weights = get_batch(train_set, BATCH_SIZE)
            [loss, _] = sess.run([train_loss, update_step], feed_dict={
                            encoder_inputs:batch_encoder_inputs, encoder_lengths:batch_encoder_lengths, decoder_inputs:batch_decoder_inputs,
                            decoder_lengths:batch_decoder_lengths, decoder_outputs:batch_decoder_outputs, target_weights:batch_target_weights})
            if step % PRINT_STEPS == 0:
                print(time.ctime(), 'step=', step, 'loss=', loss, 'learning_rate=', LEARNING_RATE)
                # 模型持久化
                saver.save(sess, MODEL_PATH+MODEL_NAME, global_step=step)
            if step % LEARNING_DECAY_STEPS == 0:
                LEARNING_RATE *= LEARNING_DECAY_RATE


def inference():
    """
    预测过程
    """
    encoder_inputs = tf.placeholder(tf.int32, shape=[1, None], name='encoder_inputs')
    encoder_lengths = tf.placeholder(tf.int32, shape=(1,), name='encoder_inputs_length')
    answers = model(encoder_inputs, encoder_lengths, vocab_size)
    
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint file found")
            return

        sys.stdout.write("> ")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()
        while input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list(input_seq)
            sample_encoder_inputs = [[n for n in input_id_list]]
            if (len(input_id_list)):
                outputs_seq = sess.run(answers, feed_dict={encoder_inputs:sample_encoder_inputs, encoder_lengths:[len(input_id_list)]})
                outputs_seq = outputs_seq.reshape((-1))
                res = ''
                for n in outputs_seq:
                    if n==EOS_ID:
                        print(res)
                    else:
                        res += wordToken.id2word(n)
            else:
                print("WARN：词汇不在服务区")

            sys.stdout.write("> ")
            sys.stdout.flush()
            input_seq = sys.stdin.readline()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        inference()
