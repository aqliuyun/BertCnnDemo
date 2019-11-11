import json
import keras_bert
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras_bert import Tokenizer

Train = False
max_word_length = 512

# define the model
def BertTextCnn(base_model,count):
    output = base_model.output#512*768
    output = keras.layers.Lambda(lambda x: x)(output)
    
    output = keras.layers.Conv1D(32,2,activation = 'tanh')(output)
    output = keras.layers.AveragePooling1D(2,strides=1)(output)
    output = keras.layers.Conv1D(64,3,activation = 'tanh')(output)
    output = keras.layers.AveragePooling1D(2,strides=1)(output)
    output = keras.layers.Conv1D(64,4,activation = 'tanh')(output)
    output = keras.layers.AveragePooling1D(4,strides=1)(output)
    
    output = keras.layers.Flatten()(output)
    output_y = keras.layers.Dense(count, activation='softmax')(output) #new softmax layer
    model = keras.Model(base_model.input, output_y)
    # summarize the model
    model.summary()
    return model

checkpoint_paths = keras_bert.get_checkpoint_paths('./chinese_L-12_H-768_A-12')
token_dict = keras_bert.loader.load_vocabulary(checkpoint_paths.vocab)
tokenizer = keras_bert.tokenizer.Tokenizer(token_dict)


# define documents
max_labels = 0;
x_tokens = []
x_segments = []
y = []
labels = []
with open('./datas/questions.json') as fp:
    loaded_json = json.load(fp)    
    for doc in loaded_json:        
        labels.append(doc['label'])        
        for q in doc['questions']:    
            x_token,x_segment = tokenizer.encode(q,max_len = max_word_length)
            x_tokens.append(x_token)
            x_segments.append(x_segment)
            y.append(max_labels)
        max_labels+=1
y = keras.utils.to_categorical(y, max_labels)


if not Train:
    bert_model,config = keras_bert.loader.build_model_from_config(checkpoint_paths.config,training=False)
    model = BertTextCnn(bert_model,len(labels))
    model.load_weights('./models/qnamaker-1.00.hdf5')
else:
    #bert model
    bert_model = keras_bert.loader.load_trained_model_from_checkpoint(checkpoint_paths.config,checkpoint_paths.checkpoint,False)
    for l in bert_model.layers:
        l.trainable = False

while not Train:
    print('input:')
    text = input()
    predict_x,segment_x = tokenizer.encode(text,max_len = max_word_length)
    scores = model.predict([[predict_x],[segment_x]])[0]    
    topindex = np.argmax(model.predict([[predict_x],[segment_x]]))
    print(labels[topindex])
    print(scores[topindex])

earlyStopping = keras.callbacks.EarlyStopping(monitor='acc', patience=8, verbose=0, mode='auto')
output_model_file = './models/qnamaker-{acc:.2f}.hdf5'
savecheckpoint = keras.callbacks.ModelCheckpoint(output_model_file, monitor='acc', verbose=1, save_best_only=True)


model = BertTextCnn(bert_model,max_labels)
# compile the model
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['acc'])
# fit the model
model.fit([x_tokens,x_segments], y, epochs=50, verbose=2,callbacks=[earlyStopping,savecheckpoint])

# evaluate the model
loss, accuracy = model.evaluate([x_tokens,x_segments], y, verbose=0)
print('Accuracy: %f' % (accuracy*100))
