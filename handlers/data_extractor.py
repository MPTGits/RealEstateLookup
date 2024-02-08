import requests
import json
import pandas as pd

neighnourhoods = json.loads(requests.get('https://www.homes.bg/api/neighbourhoods?locationId=1&agencyId=&typeId=ApartmentSell').text)

for (id, name) in neighnourhoods:
    result_data = []
    for i in range(0, 10):
        startIndex = i * 100
        stopIndex = startIndex+100
        data = json.loads(requests.get(f'https://www.homes.bg/api/offers?neighbourhoods[]={int(id)}&startIndex={startIndex}&stopIndex={stopIndex}').text).get('result')
        for idx, offer in enumerate(data):
            offer = json.loads(requests.get(f'https://www.homes.bg/api/offers/as/{offer.get("id")}').text).get('data', {})
            specification = {}
            title = offer.get('title', '').split(',')
            specification['Размер'] = title[1].strip() if len(title) > 1 else ''
            specification['Адрес'] = offer.get('address', {}).get("city", '')
            specification['Точен адрес'] = bool(offer.get('address', {}).get("is_exact_address", ''))
            specification['Цена'] = offer.get('price', {}).get('value', '')
            specification['Валута'] = offer.get('price', {}).get('currency', '')
            offer_attrs = offer.get('attributes')
            for attr in offer_attrs:
                if not attr.get('label'):
                    specification['Oписание'] = attr.get('value')
                else:
                    specification[attr.get('label')] = attr.get('value', '')
            result_data.append(specification)
            print(specification)
            print(startIndex+idx)

    df = pd.DataFrame(result_data)

    # Write the DataFrame to a CSV file
    df.to_csv(f'real_estate_{name}.csv', index=False)
