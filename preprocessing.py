import pandas as pd
import spacy


def preprocessing_text(text):
    nlp=spacy.load('en_core_web_sm')
    
    nlp=nlp(str(text).lower())
    tokens=[]
    for row in nlp:
        if not row.is_stop and not row.is_punct:
            tokens.append(row.lemma_)
    return " ".join(tokens)


def clean_data(datapath):
    
    df=pd.read_csv(datapath)
    
    combined_text=(df['about_me']
    + " " + df['professional_summary']
    + " " + df['interests'] +" "+ df['skills']
    + " " +df['career_goals']+ " " +df['work_style'] + " "
    + df['personal_values'])
    
    df['cleaned_text']=combined_text.apply(preprocessing_text)
    
    df.to_csv("cleaned_dataset.csv",index=False)

if __name__=='__main__':
    datapath="dataset.csv"
    clean_data(datapath)