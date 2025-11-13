# stocks/views.py
from django.core.cache import cache
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

from stock.models import WatchList, StockInfo
from stock.serializers import WatchListSerializer, StockInfoSerializer


class WatchListCreateListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]  # adjust to your auth needs

    def get(self, request):
        watchlists = WatchList.objects.filter(user=request.user)
        serializer = WatchListSerializer(watchlists, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    def post(self, request):
        serializer = WatchListSerializer(data = {
            "name": request.data.get('name'),
            "user": request.user.id,
        })
        if serializer.is_valid():
            watchlist = serializer.save()
            return Response(WatchListSerializer(watchlist).data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request):
        watchlist_id = request.data.get('watchlist_id')
        if not watchlist_id:
            return Response({"detail": "Field watchlist_id needs to be provided."}, status=status.HTTP_400_BAD_REQUEST)
        watchlist = get_object_or_404(WatchList, user=request.user, id=watchlist_id)
        watchlist.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class WatchListDetailAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]  # adjust to your auth needs

    def get(self, request):
        watchlist_id = request.query_params.get('watchlist_id')
        watchlist = get_object_or_404(WatchList, user=request.user, id=watchlist_id)
        return Response({
            "name": watchlist.name,
            "stock": StockInfoSerializer(watchlist.stocks, many=True).data
        })


class AddToWatchListAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]  # adjust to your auth needs

    def post(self, request):
        watchlist_id = request.data.get('watchlist_id')
        if not watchlist_id:
            return Response({"detail": "Field watchlist_id needs to be provided."}, status=status.HTTP_400_BAD_REQUEST)
        watchlist = get_object_or_404(WatchList, user=request.user, id=watchlist_id)
        stock, _ = StockInfo.objects.get_or_create(symbol=request.data.get('symbol'))
        watchlist.stocks.add(stock)
        return Response(WatchListSerializer(watchlist).data, status=status.HTTP_201_CREATED)


class StockInfoDetailAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        stocks = StockInfo.objects.all()
        stock_price = cache.get('stock:realtime_price')
        return Response({"stocks": StockInfoSerializer(stocks, many=True).data, "stock_price": stock_price})

    def delete(self, request):
        symbol = request.data.get('symbol')
        stocks = StockInfo.objects.filter(symbol=symbol)
        stocks.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
