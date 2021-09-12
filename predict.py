import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('idxtoword',"rb") as fp:
    idxtoword = pickle.load(fp)

with open('wordtoidx',"rb") as fp:
    wordtoidx = pickle.load(fp)


caption_model.load_weights("fresh_model_wt.hdf5")
max_length = 34
OUTPUT_DIM = 2048

def generateCaption(photo):
    in_text = 'startseq'
    for i in range(max_length): #The maximum length of caption cannot exceed max_length = 34
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == END:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

for z in range(5): # set higher to see more examples
    pic = list(encoding_test.keys())[z]
    image = encoding_test[pic].reshape((1,OUTPUT_DIM))
    print(os.path.join(root_dir,'Flicker8k_Dataset', pic))
    x=plt.imread(os.path.join(root_dir,'Flicker8k_Dataset', pic))
    plt.imshow(x)
    plt.show()
    print("Caption:",generateCaption(image))
    print("_____________________________________")
