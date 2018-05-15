# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    I replace this with an LSTM, leads to 2% improvement
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.rnn_cell_fw = rnn_cell.LSTMCell(value_vec_size/2, reuse=tf.AUTO_REUSE)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(value_vec_size/2, reuse=tf.AUTO_REUSE)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, values, values_mask, keys):
        """
        Blended representation consists of keys and the parts of the values relevant to the keys
        """
        _, attn_output = compute_attention(self.keep_prob, values, values_mask, keys) # attn_output is shape (batch_size, context_len, hidden_size*2)

        # Concat attn_output to context_hiddens to get blended_reps
        blended_reps = tf.concat([keys, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)
        with tf.variable_scope('BasicAttn_BRNN', reuse=True):
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, blended_reps, dtype=tf.float32,
                                                                  scope='BasicAttn_BRNN')

        # Concatenate the forward and backward hidden states
        blended_reps_dense = tf.concat([fw_out, bw_out], 2)
        # Apply dropout
        blended_reps_final = tf.nn.dropout(blended_reps_dense, self.keep_prob)
        return blended_reps_final

class CoAttn_simplified(object):
    """
    Module for coattention, only the barebones of coattention are implemented : Q2C attention, C2Q attention, coattention which is then fed
    into a fully_connected layer
    Uses an affinity matix to compute the attention scores
    """
    def __init__(self, keep_prob, qn_vec_size, cxt_vec_size):
        """
        Inputs:
            keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
            qn_vec_size: size of the question vectors. int
            cxt_vec_size: size of the context vectors. int
        """
        self.keep_prob = keep_prob
        self.qn_vec_size = qn_vec_size
        self.cxt_vec_size = cxt_vec_size

    def build_graph(self, qn_hiddens, cxt_hiddens, qn_mask, cxt_mask):
        """
        Compute C2Q attention, Q2C attention and then coattention
        """
        question_len = 30
        context_len = 450
        # Always using batch_size = 128 does not work, since the number of examples in the validation set is unlikely to be a multiple of
        # 128. The final batch would have to have <128 examples unlike in training where batches get filled up fully each time.
        batch_size = tf.shape(qn_hiddens)[0]
        qn_sentinel = tf.tile(tf.zeros([1,1,self.qn_vec_size]), [batch_size, 1, 1])
        # qn_hiddens has shape (batch_size, qn_len, qn_vec_size)
        qn_hiddens = tf.concat([qn_hiddens, qn_sentinel], axis=1)
        # qn_hiddens has shape (batch_size, qn_len + 1, qn_vec_size)
        cxt_sentinel = tf.tile(tf.zeros([1,1,self.cxt_vec_size]), [batch_size, 1, 1])
        # cxt_hiddens has shape (batch_size, context_len, cxt_vec_size)
        cxt_hiddens = tf.concat([cxt_hiddens, cxt_sentinel], axis=1)
        # cxt_hiddens has shape (batch_size, context_len + 1, cxt_vec_size)

        # Add a non-linear projection layer to allow for variation between the context encoding space and the question encoding space
        qn_hiddens = tf.contrib.layers.fully_connected(qn_hiddens, num_outputs=self.qn_vec_size, activation_fn=tf.nn.tanh)
        # qn_hiddens still has the shape (batch_size, qn_len + 1, qn_vec_size)

        # build the affinity matrix
        qn_hiddens_t = tf.transpose(qn_hiddens, perm=[0,2,1])
        L = tf.matmul(cxt_hiddens, qn_hiddens_t) # L shape : (batch_size, num_cxt + 1, num_qn + 1)
        # Obtain C2Q attention outputs
        # Unmask the sentinel vector
        alpha = tf.nn.softmax(L) # alpha shape : (batch_size, num_cxt + 1, num_qn + 1)
        a = tf.matmul(alpha, qn_hiddens)    # a shape : (batch_size, num_cxt, qn_vec_size)
        alpha_n = tf.expand_dims(alpha,-1) # alpha_n shape : (batch_size, num_cxt, num_qn, 1)
        # Obtain C2Q attention outputs
        L_t = tf.transpose(L, perm=[0,2,1])
        beta = tf.nn.softmax(L_t) # beta shape : (batch_size, num_cxt+1, num_qn+1)
        b = tf.matmul(beta, cxt_hiddens)    # b shape : (batch_size, num_qn, cxt_vec_size)
        b_n = tf.expand_dims(b, 1)
        coattnCxt = tf.reduce_sum(alpha_n * b_n, 2)
        # alpha_n*b shape : (batch_size, num_cxt, num_qn, cxt_vec_size) coattnCxt shape : (batch_size, num_cxt, cxt_vec_size)
        blended_reps = tf.concat([coattnCxt, a], axis=2) # shape : (batch_size, num_cxt, 2*vec_size)
        rnn_cell_fw = rnn_cell.LSTMCell(self.cxt_vec_size/2)
        rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
        rnn_cell_bw = rnn_cell.LSTMCell(self.cxt_vec_size/2)
        rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, blended_reps, dtype=tf.float32)

        # Concatenate the forward and backward hidden states
        blended_reps_dense = tf.concat([fw_out, bw_out], 2)
        # Apply dropout
        blended_reps_final = tf.nn.dropout(blended_reps_dense, self.keep_prob)
        return blended_reps_final

class CoAttn_zeroSentinel(object):
    """
    Module for coattention, only the barebones of coattention are implemented : Q2C attention, C2Q attention, coattention which is then fed
    into a fully_connected layer
    Uses an affinity matix to compute the attention scores
    """
    def __init__(self, keep_prob, qn_vec_size, cxt_vec_size):
        """
        Inputs:
            keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
            qn_vec_size: size of the question vectors. int
            cxt_vec_size: size of the context vectors. int
        """
        self.keep_prob = keep_prob
        self.qn_vec_size = qn_vec_size
        self.cxt_vec_size = cxt_vec_size

    def build_graph(self, qn_hiddens, cxt_hiddens, qn_mask, cxt_mask):
        """
        Compute C2Q attention, Q2C attention and then coattention
        """
        question_len = 30
        context_len = 450
        # Always using batch_size = 128 does not work, since the number of examples in the validation set is unlikely to be a multiple of
        # 128. The final batch would have to have <128 examples unlike in training where batches get filled up fully each time.
        batch_size = tf.shape(qn_hiddens)[0]
        # Match dimensions qith qn_hiddens for concatenation
        qn_tiled = tf.tile(tf.zeros([1,1,self.qn_vec_size]), [batch_size, 1, 1])
        # qn_hiddens has shape (batch_size, qn_len, qn_vec_size)
        # qn_tiled has shape (batch_size, 1, qn_vec_size)
        qn_hiddens = tf.concat([qn_hiddens, qn_tiled], axis=1)
        # qn_hiddens has shape (batch_size, qn_len + 1, qn_vec_size)

        # Match dimensions qith cxt_hiddens for concatenation
        cxt_tiled = tf.tile(tf.zeros([1,1,self.cxt_vec_size]), [batch_size, 1, 1])
        # cxt_hiddens has shape (batch_size, context_len, cxt_vec_size)
        cxt_hiddens = tf.concat([cxt_hiddens, cxt_tiled], axis=1)
        # cxt_hiddens has shape (batch_size, context_len + 1, cxt_vec_size)

        # Add a non-linear projection layer to allow for variation between the context encoding space and the question encoding space
        qn_hiddens = tf.contrib.layers.fully_connected(qn_hiddens, num_outputs=self.qn_vec_size, activation_fn=tf.nn.tanh)
        # qn_hiddens still has the shape (batch_size, qn_len + 1, qn_vec_size)

        # build the affinity matrix
        qn_hiddens_t = tf.transpose(qn_hiddens, perm=[0,2,1])
        L = tf.matmul(cxt_hiddens, qn_hiddens_t) # L shape : (batch_size, num_cxt + 1, num_qn + 1)
        # Obtain C2Q attention outputs
        # Unmask the sentinel vector
        qn_mask = tf.concat([qn_mask, tf.ones([batch_size, 1], tf.int32)], axis=1)
        alpha_logits_mask = tf.expand_dims(qn_mask, 1) # shape (batch_size, 1, num_qn + 1)
        _, alpha = masked_softmax(L, alpha_logits_mask, 2) # alpha shape : (batch_size, num_cxt + 1, num_qn + 1)
        # The purpose of the sentinel vector is to allow the model to decide not to pay attention to any value, so it can be discarded after
        # computing the softmax probabilities.
        alpha = alpha[:,0:-1,0:-1]  # Remove any probabilities which involve the sentinels
        # alpha shape : (batch_size, num_cxt, num_qn)
        qn_hiddens = qn_hiddens[:,0:-1,:]  # Remove the sentinel from the hidden vectors
        # qn_hiddens has the shape (batch_size, qn_len, qn_vec_size)
        a = tf.matmul(alpha, qn_hiddens)    # a shape : (batch_size, num_cxt, qn_vec_size)
        alpha_n = tf.expand_dims(alpha,-1) # alpha_n shape : (batch_size, num_cxt, num_qn, 1)
        # Obtain C2Q attention outputs
        L_t = tf.transpose(L, perm=[0,2,1])
        cxt_mask = tf.concat([cxt_mask, tf.ones([batch_size, 1], dtype=tf.int32)], axis=1)
        beta_logits_mask = tf.expand_dims(cxt_mask, 1) # shape (batch_size, 1, num_cxt+1)
        _, beta = masked_softmax(L_t, beta_logits_mask, 2) # beta shape : (batch_size, num_cxt+1, num_qn+1)
        beta = beta[:,0:-1,0:-1] # shape : (batch_size, num_cxt, num_qn)
        cxt_hiddens = cxt_hiddens[:,0:-1,:] # shape : (batch_size, context_len, cxt_vec_size)
        b = tf.matmul(beta, cxt_hiddens)    # b shape : (batch_size, num_qn, cxt_vec_size)
        b_n = tf.expand_dims(b, 1)
        coattnCxt = tf.reduce_sum(alpha_n * b_n, 2)
        # alpha_n*b shape : (batch_size, num_cxt, num_qn, cxt_vec_size) coattnCxt shape : (batch_size, num_cxt, cxt_vec_size)
        blended_reps = tf.concat([coattnCxt, a], axis=2) # shape : (batch_size, num_cxt, 2*vec_size)
        rnn_cell_fw = rnn_cell.LSTMCell(self.cxt_vec_size/2)
        rnn_cell_fw = DropoutWrapper(rnn_cell_fw, input_keep_prob=self.keep_prob)
        rnn_cell_bw = rnn_cell.LSTMCell(self.cxt_vec_size/2)
        rnn_cell_bw = DropoutWrapper(rnn_cell_bw, input_keep_prob=self.keep_prob)
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, blended_reps, dtype=tf.float32)

        # Concatenate the forward and backward hidden states
        blended_reps_dense = tf.concat([fw_out, bw_out], 2)
        # Apply dropout
        blended_reps_final = tf.nn.dropout(blended_reps_dense, self.keep_prob)
        return blended_reps_final

class CoAttn(object):
    """
    Module for coattention, only the barebones of coattention are implemented : Q2C attention, C2Q attention, coattention which is then fed
    into a fully_connected layer
    Uses an affinity matix to compute the attention scores
    """
    def __init__(self, keep_prob, qn_vec_size, cxt_vec_size):
        """
        Inputs:
            keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
            qn_vec_size: size of the question vectors. int
            cxt_vec_size: size of the context vectors. int
        """
        self.keep_prob = keep_prob
        self.qn_vec_size = qn_vec_size
        self.cxt_vec_size = cxt_vec_size
        self.rnn_cell_fw = rnn_cell.LSTMCell(cxt_vec_size/2, reuse=tf.AUTO_REUSE)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.LSTMCell(cxt_vec_size/2, reuse=tf.AUTO_REUSE)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, qn_hiddens, cxt_hiddens, qn_mask, cxt_mask):
        """
        Compute C2Q attention, Q2C attention and then coattention
        """
        question_len = 30
        context_len = 450
        # Always using batch_size = 128 does not work, since the number of examples in the validation set is unlikely to be a multiple of
        # 128. The final batch would have to have <128 examples unlike in training where batches get filled up fully each time.
        batch_size = tf.shape(qn_hiddens)[0]
        # Add sentinel vectors to both the context and the question hidden states to allow the model to not attend to any word in the input
        with tf.variable_scope("sentinel", reuse=tf.AUTO_REUSE):
            qn_sentinel = tf.get_variable("qn_sentinel",[1, 1, self.qn_vec_size])
            cxt_sentinel = tf.get_variable("cxt_sentinel",[1, 1, self.cxt_vec_size])

        # Match dimensions qith qn_hiddens for concatenation
        qn_tiled = tf.tile(qn_sentinel, [batch_size, 1, 1])
        # qn_hiddens has shape (batch_size, qn_len, qn_vec_size)
        qn_hiddens = tf.concat([qn_hiddens, qn_tiled], axis=1)
        # qn_hiddens has shape (batch_size, qn_len + 1, qn_vec_size)

        # Match dimensions qith cxt_hiddens for concatenation
        cxt_tiled = tf.tile(cxt_sentinel, [batch_size, 1, 1])
        # cxt_hiddens has shape (batch_size, context_len, cxt_vec_size)
        cxt_hiddens = tf.concat([cxt_hiddens, cxt_tiled], axis=1)
        # cxt_hiddens has shape (batch_size, context_len + 1, cxt_vec_size)

        # Add a non-linear projection layer to allow for variation between the context encoding space and the question encoding space
        qn_hiddens = tf.contrib.layers.fully_connected(qn_hiddens, num_outputs=self.qn_vec_size, activation_fn=tf.nn.tanh)
        # qn_hiddens still has the shape (batch_size, qn_len + 1, qn_vec_size)
#        W_Q = tf.get_variable("W_downProjection", [question_len, question_len + 1])
#        b_Q = tf.get_variable("bias_downProjection", [1, qn_vec_size])
#        qn_hiddens = tf.tanh(tf.matmul(W_Q, qn_hiddens) + b_Q)

        # build the affinity matrix
        qn_hiddens_t = tf.transpose(qn_hiddens, perm=[0,2,1])
        L = tf.matmul(cxt_hiddens, qn_hiddens_t) # L shape : (batch_size, num_cxt + 1, num_qn + 1)
        # Obtain C2Q attention outputs
        # Unmask the sentinel vector
        qn_mask = tf.concat([qn_mask, tf.ones([batch_size, 1], tf.int32)], axis=1)
        alpha_logits_mask = tf.expand_dims(qn_mask, 1) # shape (batch_size, 1, num_qn + 1)
        _, alpha = masked_softmax(L, alpha_logits_mask, 2) # alpha shape : (batch_size, num_cxt + 1, num_qn + 1)
        # The purpose of the sentinel vector is to allow the model to decide not to pay attention to any value, so it can be discarded after
        # computing the softmax probabilities.
        alpha = alpha[:,0:-1,0:-1]  # Remove any probabilities which involve the sentinels
        # alpha shape : (batch_size, num_cxt, num_qn)
        qn_hiddens = qn_hiddens[:,0:-1,:]  # Remove the sentinel from the hidden vectors
        # qn_hiddens has the shape (batch_size, qn_len, qn_vec_size)
        a = tf.matmul(alpha, qn_hiddens)    # a shape : (batch_size, num_cxt, qn_vec_size)
        alpha_n = tf.expand_dims(alpha,-1) # alpha_n shape : (batch_size, num_cxt, num_qn, 1)
        # Obtain C2Q attention outputs
        L_t = tf.transpose(L, perm=[0,2,1])
        cxt_mask = tf.concat([cxt_mask, tf.ones([batch_size, 1], dtype=tf.int32)], axis=1)
        beta_logits_mask = tf.expand_dims(cxt_mask, 1) # shape (batch_size, 1, num_cxt+1)
        _, beta = masked_softmax(L_t, beta_logits_mask, 2) # beta shape : (batch_size, num_cxt+1, num_qn+1)
        beta = beta[:,0:-1,0:-1] # shape : (batch_size, num_cxt, num_qn)
        cxt_hiddens = cxt_hiddens[:,0:-1,:] # shape : (batch_size, context_len, cxt_vec_size)
        b = tf.matmul(beta, cxt_hiddens)    # b shape : (batch_size, num_qn, cxt_vec_size)
        b_n = tf.expand_dims(b, 1)
        coattnCxt = tf.reduce_sum(alpha_n * b_n, 2)
        # alpha_n*b shape : (batch_size, num_cxt, num_qn, cxt_vec_size) coattnCxt shape : (batch_size, num_cxt, cxt_vec_size)
        blended_reps = tf.concat([coattnCxt, a], axis=2) # shape : (batch_size, num_cxt, 2*vec_size)
        (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, blended_reps, dtype=tf.float32)

        # Concatenate the forward and backward hidden states
        blended_reps_dense = tf.concat([fw_out, bw_out], 2)
        # Apply dropout
        blended_reps_final = tf.nn.dropout(blended_reps_dense, self.keep_prob)
        return blended_reps_final

def compute_attention(keep_prob, values, values_mask, keys):
    """
    Keys attend to values.
    For each key, return an attention distribution and an attention output vector.

    Inputs:
        keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
        values: Tensor shape (batch_size, num_values, value_vec_size).
        values_mask: Tensor shape (batch_size, num_values).
        1s where there's real input, 0s where there's padding
        keys: Tensor shape (batch_size, num_keys, value_vec_size)

    Outputs:
        attn_dist: Tensor shape (batch_size, num_keys, num_values).
        For each key, the distribution should sum to 1,
        and should be 0 in the value locations that correspond to padding.
        output: Tensor shape (batch_size, num_keys, hidden_size).
        This is the attention output; the weighted sum of the values
        (using the attention distribution as weights).
    """
    # Calculate attention distribution
    values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
    attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
    attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
    _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

    # Use attention distribution to take weighted sum of values
    output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

    # Apply dropout
    output = tf.nn.dropout(output, keep_prob)

    return attn_dist, output

def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
