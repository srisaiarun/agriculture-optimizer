
from django import forms
from django.core.validators import MinValueValidator
from .models import FarmerProblem

CROP_CHOICES = [
    ('wheat', 'Wheat'),
    ('rice', 'Rice'),
    ('corn', 'Corn'),
    ('soybean', 'Soybean'),
    ('barley', 'Barley'),
    ('cotton', 'Cotton'),
    ('sugarcane', 'Sugarcane'),
]

class AgricultureForm(forms.Form):
    # Resource constraints
    land_available = forms.FloatField(
        label="Total Land Available (acres)",
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 100.5'
        })
    )
    
    labor_available = forms.FloatField(
        label="Total Labor Available (hours)",
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 2000'
        })
    )
    
    # First crop details
    crop1 = forms.ChoiceField(
        label="First Crop",
        choices=CROP_CHOICES,
        initial='wheat',
        widget=forms.Select(attrs={'class': 'crop-select'})
    )
    
    profit_crop1 = forms.FloatField(
        label="Profit per Acre (₹)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 5000'
        })
    )
    
    labor_crop1 = forms.FloatField(
        label="Labor Required per Acre (hours)",
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 20'
        })
    )
    
    # Second crop details
    crop2 = forms.ChoiceField(
        label="Second Crop",
        choices=CROP_CHOICES,
        initial='rice',
        widget=forms.Select(attrs={'class': 'crop-select'})
    )
    
    profit_crop2 = forms.FloatField(
        label="Profit per Acre (₹)",
        min_value=0,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 7000'
        })
    )
    
    labor_crop2 = forms.FloatField(
        label="Labor Required per Acre (hours)",
        min_value=0.01,
        widget=forms.NumberInput(attrs={
            'step': '0.01',
            'placeholder': 'e.g., 30'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        crop1 = cleaned_data.get('crop1')
        crop2 = cleaned_data.get('crop2')
        
        if crop1 == crop2:
            raise forms.ValidationError("Please select two different crops.")
        
        # Validate that labor doesn't exceed available resources
        labor_crop1 = cleaned_data.get('labor_crop1')
        labor_crop2 = cleaned_data.get('labor_crop2')
        land_available = cleaned_data.get('land_available')
        
        if labor_crop1 and labor_crop2 and land_available:
            min_labor_required = min(labor_crop1, labor_crop2) * land_available
            labor_available = cleaned_data.get('labor_available')
            
            if labor_available and min_labor_required > labor_available:
                raise forms.ValidationError(
                    f"With these labor requirements, you need at least {min_labor_required:.2f} "
                    f"hours of labor for {land_available} acres, but only have {labor_available}."
                )
        
        return cleaned_data
    
    



class FarmerProblemForm(forms.ModelForm):
    class Meta:
        model = FarmerProblem
        fields = [
            'crop1_name', 'crop2_name',
            'crop1_profit', 'crop2_profit',
            'total_land',
            'labor_available', 'crop1_labor', 'crop2_labor',
            'water_available', 'crop1_water', 'crop2_water'
        ]
        widgets = {
            'crop1_name': forms.TextInput(attrs={'class': 'form-control'}),
            'crop2_name': forms.TextInput(attrs={'class': 'form-control'}),
            'crop1_profit': forms.NumberInput(attrs={'class': 'form-control'}),
            'crop2_profit': forms.NumberInput(attrs={'class': 'form-control'}),
            'total_land': forms.NumberInput(attrs={'class': 'form-control'}),
            'labor_available': forms.NumberInput(attrs={'class': 'form-control'}),
            'crop1_labor': forms.NumberInput(attrs={'class': 'form-control'}),
            'crop2_labor': forms.NumberInput(attrs={'class': 'form-control'}),
            'water_available': forms.NumberInput(attrs={'class': 'form-control'}),
            'crop1_water': forms.NumberInput(attrs={'class': 'form-control'}),
            'crop2_water': forms.NumberInput(attrs={'class': 'form-control'}),
        }