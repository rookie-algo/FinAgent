from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

from users.views import UserViewSet
from stock.views import AddToWatchListAPIView, WatchListCreateListAPIView, \
    WatchListDetainAPIView, StockInfoDetailAPIView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    # Authorization
    path("api/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # Get or Add user
    path('api/users/', UserViewSet.as_view({'get': 'list', 'post': 'create'}), name='get-users'),
    path('api/watchlists/', WatchListCreateListAPIView.as_view(), name='watchlist-list-create'),
    path('api/watchlists/detail/', WatchListDetainAPIView.as_view(), name='watchlist-list-create'),
    
    path('api/add-to-list/', AddToWatchListAPIView.as_view(), name='watchlist-list-create'),
    path('api/stock-info/', StockInfoDetailAPIView.as_view(), name='watchlist-list-create'),
    
]
