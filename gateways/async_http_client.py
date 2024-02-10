import aiohttp

class AsyncHTTPClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def request(self, method, url, params=None, data=None):
        async with self.session.request(method, self.base_url+url, params=params, json=data) as response:
            return await response.json()
