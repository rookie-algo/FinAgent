from django.conf import settings
from rest_framework.exceptions import ValidationError
from django.db import models
import yfinance as yf

TIER_LIMITS = {'free': 5, 'pro': 50, 'expert': 200}


class StockInfo(models.Model):
    symbol = models.CharField(max_length=50, unique=True)
    industry = models.CharField(max_length=50, blank=True)
    website = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=20, blank=True)
    summary = models.CharField(max_length=1000, blank=True)

    def save(self, *args, **kwargs):
        # Only fetch data when creating (not updating)
        if not self.pk:
            ticker = yf.Ticker(self.symbol.upper())
            
            info = ticker.info  # may trigger a web request
            if info and 'regularMarketPrice' in ticker.info:
                self.industry = info.get("industry", "")
                self.website = info.get("website", "")
                self.country = info.get("country", "")
                self.summary = info.get("longBusinessSummary", "")
            else:
                raise ValidationError({'detail': f'Symbol {self.symbol.upper()} does not exists'})

        super().save(*args, **kwargs)


class WatchList(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='watchlists',
    )
    name = models.CharField(max_length=100)
    created = models.DateTimeField(auto_now_add=True)

    stocks = models.ManyToManyField(
        StockInfo,
        related_name='in_watchlists',
        blank=True,
    )

    def __str__(self):
        return f"{self.name} ({self.user.email})"

    def clean(self):
        """
        Model-level guard (useful when saving via admin or calling .full_clean()).
        Note: For REST writes, we also enforce in a signal (below) and/or serializer.
        """
        super().clean()
        if not self.user_id:
            return
        max_allowed = TIER_LIMITS.get(self.user.tier, 5)
        if self.pk and self.stocks.count() > max_allowed:
            raise ValidationError({'stocks': f'{self.user.tier.capitalize()} tier allows max {max_allowed} symbols.'})