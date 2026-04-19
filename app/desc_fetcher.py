from typing import Literal
import wikipediaapi
import asyncio
wiki = wikipediaapi.AsyncWikipedia("Dr. Akinator")

async def getSymptomDesc(symptom:str) -> tuple[str, Literal['hard_coded', 'wikipedia', 'none_found']]: # (description, source)
    # some symptoms generally are just hard to get
    hardDesc:dict[str, str] = {
        'mild_fever': 'Body temperature that is slightly higher than normal, typically within the range 37.3°C–38.3°C',
        'high_fever': 'Body temperature that is significantly higher than normal, typically above 39.4°C',
        'yellow_skin': 'A yellowish pigmentation of the skin'
    }

    if symptom in hardDesc:
        return (hardDesc[symptom], 'hard_coded')

    # fallback to wikipedia api

    # wikipedia has it, but wrong term
    replacements:dict[str, str] = {
        'abnormal_menstruation': 'irregular_menstruation'
    }

    if symptom in replacements: symptom = replacements[symptom]
    page = wiki.page(symptom.replace('_', ' ').title())
    if await page.exists():
        desc = await page.summary
        return (desc, 'wikipedia')

    return ("No description available", 'none_found')

if __name__ == "__main__":
    res = asyncio.run(getSymptomDesc('vomiting'))
    print(res)
