from requests import Request

from constants import HTTPMethod


class GetOffersMetadataHandler:
    """"
    Query strings:
        - startIndex(int) - index of the first offer to get
        - stopIndex (int) - index of the last offer to get
    Returns metadata for offers in the form:
    "result": [
        {
            "id": "1386651",
            "type": "as",
            "viewHref": "/offer/apartament-za-prodazhba/dvustaen-92m2-sofiya-kv.-vitosha/as1386651",
            "time": "вчера",
            "location": "кв. Витоша, София",
            "title": "Двустаен, 92m²",
            "description": "Тухла",
            "status": 1,
            "photos": [
                {
                    "name": "93934161",
                    "path": "2023-01-14_2/",
                    "width": 252,
                    "height": 336
                }
            ],
            "price": {
                "value": "149,000",
                "currency": "EUR",
                "price_per_square_meter": "1,619 EUR/m²",
                "period": null
            },
            "isFav": false
        },
        """
    PATH = '/offers'
    METHOD = HTTPMethod.GET

    def __init__(self, client):
        self.client = client


    async def __call__(self, request: Request):
        offers_data = self.client.request(self.METHOD, self.PATH, request.params)
        return await self.extract_offers_path(offers_data.get('results'))

    async def extract_offers_path(self, offers_data):
        offers_path = []
        for offer_data in offers_data:
            offers_path.append(offer_data.get('viewHref'))
        return offers_path
