import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, LSTM, Softmax, \
    Lambda, Average, Dot, Permute, Multiply, Add, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, Concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras import backend
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate

class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs

class Attention(OurLayer):
    """多头注意力机制
    """
    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
    def call(self, inputs):
        q, k, v = inputs[: 3]
        v_mask, q_mask = None, None
        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        # a = to_mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask = (1 - K.constant(self.mask_right)) * 1e10
                mask = K.expand_dims(K.expand_dims(mask, 0), 0)
                self.mask = mask
                a = a - mask
        a = K.softmax(a)
        self.a = a
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        # o = to_mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)

# class Attn(Layer):
#
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(Attn, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.Wq = self.add_weight(shape=(input_shape[1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         self.Wk = self.add_weight(shape=(input_shape[1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         self.Wv = self.add_weight(shape=(input_shape[1], self.output_dim),
#                                   initializer='uniform',
#                                   trainable=True)
#         super(Attn, self).build(input_shape)  # Be sure to call this at the end
#
#     def call(self, x,):
#         q,k,v = x[0],K.transpose(x[1]),x[1]
#         a = K.dot(q,k)/64
#         a = k.softmax(a)
#         return K.dot(a,v)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)

class myModel():
    def __init__(self,sent_lenth,word_embedding):
        self.sent_lenth = sent_lenth
        self.word_embedding = word_embedding
        self.model = None

    def attn(self,embedding):
        sent_input = Input(shape=(self.sent_lenth,))
        sent = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="word")(sent_input)
        ent1_input = Input(shape=(5,))
        ent1 = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="ent1")(ent1_input)
        ent2_input = Input(shape=(5,))
        ent2 = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="ent2")(ent2_input)
        e1 = Attention(heads=2,size_per_head=100)([sent,ent1,ent1])
        e2 = Attention(heads=2,size_per_head=100)([sent,ent2,ent2])

        output = Multiply()([e1,e2])
        output = Dropout(0.3)(output)
        output = Flatten()(output)
        output = Dense(units=2)(output)
        # output = GRU(units=2)(output)
        output = Activation("sigmoid")(output)
        # output = Softmax()(output)
        model = Model(inputs=[sent_input,ent1_input,ent2_input],outputs=[output,])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        model.summary()
        self.model = model

    def train(self,inputs,label,save_path,validation_split, batch_size, epochs, verbose=2):
        if self.model:
            early_stopper = EarlyStopping(patience=50, verbose=1)
            check_pointer = ModelCheckpoint(save_path, verbose=1, save_best_only=True)
            print('\n---------- train begin ----------')
            self.model.fit(inputs, label,
                           validation_split=validation_split,
                           batch_size=batch_size,
                           epochs=epochs,
                           verbose=verbose,
                           callbacks=[early_stopper, check_pointer])
            print('\n---------- train done ----------')
        else:
            print("model error")

    def predict(self,inputs):
        if self.model:
            output = self.model.predict(inputs, verbose=1)
            print('\n---------- predict done ----------')
            return output
        else:
            print("model error")