from celery import shared_task
from django.core.cache import cache
import yfinance as yf

from stock.models import StockInfo


BATCH_SIZE = 15       # good for yf.Tickers


def batch_list(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def sync_all_prices(self):
    data = {}
    try:
        batches = batch_list([x.symbol for x in StockInfo.objects.all()], BATCH_SIZE)
        for symbol_list in batches:
            df = yf.download(symbol_list, period='1y', interval='1d')
            for symbol in symbol_list:
                try:
                    rk = df.xs(symbol, axis=1, level=1)
                    key = f"stock:{symbol}:history_data"
                    cache.set(key, rk)
                    data[symbol] = rk.iloc[-1][['Close', 'High', 'Low', 'Open', 'Volume']].to_dict()
                except Exception as e:
                    data[symbol] = {"error": str(e)}
    except Exception as e:
        data[symbol] = {"price": None, "error": str(e)}
    cache.set("stock:realtime_price", data)
    return data
