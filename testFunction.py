import CardRecog
import asyncio
from tensorflow import keras


def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(CardRecog.identifyimage())
    loop.close()
    return "nil"
