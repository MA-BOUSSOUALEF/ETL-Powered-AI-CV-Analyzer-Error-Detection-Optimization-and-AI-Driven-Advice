import spacy 
import pickle 
import random 

train_data = pickle.load(open("datasetCV/data_training and test/training/train_data.pkl", "rb"))
nlp = spacy.blank('en')
def train_model(train_data):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            print("Starting iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],
                        [annotations],
                        drop=0.2,
                        sgd=optimizer,
                        losses=losses
                    )
                except Exception as e:
                    pass
            print(losses)

train_model(train_data)