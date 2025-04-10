from django.db import models

class FarmerProblem(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    crop1_name = models.CharField(max_length=100, default='Wheat')
    crop2_name = models.CharField(max_length=100, default='Corn')
    crop1_profit = models.FloatField(default=5000)
    crop2_profit = models.FloatField(default=6000)
    total_land = models.FloatField(default=10)
    crop1_land = models.FloatField(blank=True, null=True)
    crop2_land = models.FloatField(blank=True, null=True)
    labor_available = models.FloatField(default=32)
    crop1_labor = models.FloatField(default=2)
    crop2_labor = models.FloatField(default=4)
    water_available = models.FloatField(default=5000)
    crop1_water = models.FloatField(default=1000)
    crop2_water = models.FloatField(default=500)
    max_profit = models.FloatField(blank=True, null=True)
    
    def __str__(self):
        return f"Problem created at {self.created_at}"