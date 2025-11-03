from rest_framework import serializers
from .models import WatchList, StockInfo, TIER_LIMITS


class WatchListSerializer(serializers.ModelSerializer):
    class Meta:
        model = WatchList
        fields = ['id', 'user', 'name', 'stocks', 'created']
        read_only_fields = ['created']

    def validate_symbols(self, value):
        user = self.context['request'].user
        max_allowed = TIER_LIMITS.get(user.tier, 10)
        if len(value) > max_allowed:
            raise serializers.ValidationError(
                f'{user.tier.capitalize()} tier allows max {max_allowed} symbols.'
            )
        return value
    

class StockInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = StockInfo
        fields = ["id", "symbol", "industry", "website", "country", "summary"]
