from logging import error
from typing import Literal
import wikipediaapi
import asyncio
import pandas as pd
import numpy as np
import csv
import nltk
nltk.download('punkt_tab')

wiki = wikipediaapi.AsyncWikipedia("Dr. Akinator")
sourceInfo = tuple[str, Literal['hard_coded', 'wikipedia', 'none_found', 'error', 'referral']]
HARD_DESC:dict[str, str] = {
    'mild_fever': 'Body temperature that is slightly higher than normal, typically within the range 37.3°C–38.3°C',
    'high_fever': 'Body temperature that is significantly higher than normal, typically above 39.4°C',
    'yellow_skin': 'A yellowish pigmentation of the skin',
    'family_history': 'Suspicion as to whether anyone in your family has suffered similar symptoms',
    'swelling_of_stomach': 'The abdomen is unusually very bloated',
    'congestion': 'Excessive fluid in tissues, vessels, or both',
    'restlessness': 'An unpleasant state of extreme arousal. Also referred to as agitation',
    'mucoid_sputum': 'A sticky mucus containing increased mucin often linked to bronchitis or asthma'
}

WIKIPEDIA_ALIAS:dict[str, str] = {
    'abnormal_menstruation': 'irregular_menstruation',
    'muscle_wasting': 'muscle_atrophy',
    'skin_rash': 'rash',
    'depression': 'Depression_(mood)',
}

async def getSymptomDesc(symptom:str) -> sourceInfo: # (description, source)
    # some symptoms generally are just hard to get
    if symptom in HARD_DESC: return (HARD_DESC[symptom], 'hard_coded')

    # fallback to wikipedia api
    # wikipedia has it, but 'wrong' term
    if symptom in WIKIPEDIA_ALIAS: symptom = WIKIPEDIA_ALIAS[symptom]
    fetched_title = symptom.replace('_', ' ').title()
    print(fetched_title)
    page = wiki.page(fetched_title)
    if await page.exists():
        desc:str = await page.summary

        if desc.endswith('may refer to:'): return (f'Referral page of {symptom}!', 'referral')
        return (' '.join([x.strip() for x in nltk.sent_tokenize(desc)[:2]] + ['(Wikipedia)']), 'wikipedia')



    return ("No description available", 'none_found')

async def main():
    disease2SympData = pd.read_csv("data/raw/dataset.csv")
    symptoms:set[str] = set([x.strip() for x in pd.unique(np.array(disease2SympData[disease2SympData.columns[1:]].values).flatten()) if isinstance(x, str)])
    symptom_size = len(symptoms)
    with open('output/valid_desc.csv', 'w') as valid, open('output/invalid.csv', 'w') as invalid:
        valid_csv = csv.writer(valid)
        invalid_csv = csv.writer(invalid)
        results:dict[str, sourceInfo] = {} 
        cur = 1
        valid_ctr = 0
        for s in symptoms:
            print(f'{cur} of {symptom_size}')
            try:
                results[s] = await getSymptomDesc(s)
            except Exception as e:
                error(e)
                results[s] = (f'Error getting information of {s}', 'error')
            
            if results[s][1] == 'wikipedia': await asyncio.sleep(0.1) # prevent api timeout
            cur += 1

            try:
                if results[s][1] in ['error', 'referral', 'none_found']:
                    invalid_csv.writerow([s, str(results[s])])
                else:
                    valid_csv.writerow([s, results[s][0]])
                    valid_ctr += 1
            except Exception as e:
                error(e)
                invalid_csv.writerow([s, 'Unparsable...?'])
        print(f'{valid_ctr} of {symptom_size} valid')
    

if __name__ == "__main__":
    asyncio.run(main())
