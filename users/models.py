from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
from django.utils import timezone


TIER_LIMITS = {'free': 5, 'pro': 50, 'expert': 200}


class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("Users must have an email address")
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        return self.create_user(email, password, **extra_fields)


class User(AbstractBaseUser, PermissionsMixin):
    class Tier(models.TextChoices):
        FREE = "free", "Free"
        PRO = "pro", "Pro"
        EXPERT = "expert", "Expert"

    email = models.EmailField(unique=True)
    username = models.CharField(max_length=50, unique=True)
    first_name = models.CharField(max_length=50, blank=True)
    last_name = models.CharField(max_length=50, blank=True)
    
    # Tier and expiration
    tier = models.CharField(
        max_length=10,
        choices=Tier.choices,
        default=Tier.FREE,
    )
    tier_expiration = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Expiration date for Pro/Expert tiers; ignored for Free users."
    )

    date_joined = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return f"{self.email} ({self.get_tier_display()})"

    # --- Helper methods ---
    def has_active_paid_tier(self):
        """Return True if user has Pro/Expert and it's not expired."""
        if self.tier == self.Tier.FREE:
            return False
        if self.tier_expiration is None:
            return False
        return self.tier_expiration > timezone.now()

    def downgrade_if_expired(self):
        """Automatically downgrade if paid tier expired."""
        if self.tier != self.Tier.FREE and (
            not self.tier_expiration or self.tier_expiration <= timezone.now()
        ):
            self.tier = self.Tier.FREE
            self.tier_expiration = None
            self.save(update_fields=["tier", "tier_expiration"])
