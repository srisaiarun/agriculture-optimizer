from django.shortcuts import render, redirect
from django.conf import settings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import os
from .forms import FarmerProblemForm
from .models import FarmerProblem
from .forms import AgricultureForm

def home(request):
    """Render the home page"""
    return render(request, 'calc/home.html')

def generate_graph(A, b, crop1_acres, crop2_acres, crop1_name, crop2_name):
    """Generate and save the optimization graph visualization"""
    plt.figure(figsize=(10, 8))
    try:
        max_x = max(b[0], b[1] / A[1][0] if A[1][0] != 0 else b[0]) * 1.2
        x = np.linspace(0, max_x, 500)

        y_land = b[0] - x
        plt.plot(x, y_land, label=f'Land: x + y ≤ {b[0]}', color='blue')

        y_labor = (b[1] - A[1][0] * x) / A[1][1] if A[1][1] != 0 else np.full_like(x, b[1]/A[1][1])
        plt.plot(x, y_labor, label=f'Labor: {A[1][0]:.1f}x + {A[1][1]:.1f}y ≤ {b[1]}', color='green')

        plt.axvline(0, color='gray', linestyle='--', label='x ≥ 0')
        plt.axhline(0, color='purple', linestyle='--', label='y ≥ 0')

        y_feasible = np.minimum(y_land, y_labor)
        y_feasible = np.maximum(y_feasible, 0)
        plt.fill_between(x, 0, y_feasible, where=(y_feasible >= 0),
                         color='lightgreen', alpha=0.3, label='Feasible Region')

        plt.scatter([crop1_acres], [crop2_acres], color='red', s=100,
                    label=f'Optimal ({crop1_acres:.1f}, {crop2_acres:.1f})')

        plt.xlabel(f"Acres of {crop1_name.capitalize()} (x)")
        plt.ylabel(f"Acres of {crop2_name.capitalize()} (y)")
        plt.title(f"Optimal {crop1_name.capitalize()}-{crop2_name.capitalize()} Cultivation")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)

        static_dir = os.path.join(settings.BASE_DIR, "static")
        os.makedirs(static_dir, exist_ok=True)
        graph_path = os.path.join(static_dir, "graph.png")

        if os.path.exists(graph_path):
            os.remove(graph_path)

        plt.savefig(graph_path, bbox_inches='tight', dpi=100)
        plt.close()

    except Exception as e:
        plt.close()
        raise Exception(f"Error generating graph: {str(e)}")

def graphical_method(request):
    """Handle the optimization form and calculations"""
    if request.method == 'POST':
        form = AgricultureForm(request.POST)
        if form.is_valid():
            try:
                data = form.cleaned_data

                A = [
                    [1, 1],
                    [data['labor_crop1'], data['labor_crop2']]
                ]
                b = [data['land_available'], data['labor_available']]
                c = [-data['profit_crop1'], -data['profit_crop2']]

                result = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)])

                if result.success:
                    crop1_acres = round(result.x[0], 2)
                    crop2_acres = round(result.x[1], 2)
                    max_profit = round(-result.fun, 2)
                    crop1_profit = round(crop1_acres * data['profit_crop1'], 2)
                    crop2_profit = round(crop2_acres * data['profit_crop2'], 2)

                    solution = {
                        'crop1': {
                            'name': data['crop1'],
                            'acres': crop1_acres,
                            'profit_per_acre': data['profit_crop1'],
                            'total_profit': crop1_profit,
                            'labor_per_acre': data['labor_crop1'],
                            'total_labor': round(crop1_acres * data['labor_crop1'], 2)
                        },
                        'crop2': {
                            'name': data['crop2'],
                            'acres': crop2_acres,
                            'profit_per_acre': data['profit_crop2'],
                            'total_profit': crop2_profit,
                            'labor_per_acre': data['labor_crop2'],
                            'total_labor': round(crop2_acres * data['labor_crop2'], 2)
                        },
                        'max_profit': max_profit,
                        'resources': {
                            'land_used': round(crop1_acres + crop2_acres, 2),
                            'land_available': data['land_available'],
                            'labor_used': round(crop1_acres * data['labor_crop1'] + crop2_acres * data['labor_crop2'], 2),
                            'labor_available': data['labor_available']
                        },
                        'constraints': {
                            'land': f"x + y ≤ {data['land_available']}",
                            'labor': f"{data['labor_crop1']:.1f}x + {data['labor_crop2']:.1f}y ≤ {data['labor_available']}"
                        }
                    }

                    try:
                        generate_graph(A, b, crop1_acres, crop2_acres, data['crop1'], data['crop2'])
                    except Exception as e:
                        form.add_error(None, f"Graph generation failed: {str(e)}")
                        return render(request, "calc/graphical_method.html", {"form": form})

                    request.session['solution'] = solution
                    return redirect('graphical_result')
                else:
                    form.add_error(None, "No feasible solution exists with the given constraints")
            except Exception as e:
                form.add_error(None, f"Error in solving the optimization problem: {str(e)}")
    else:
        form = AgricultureForm()

    return render(request, "calc/graphical_method.html", {"form": form})

def graphical_result(request):
    """Display the optimization results"""
    solution = request.session.get('solution', None)

    if not solution:
        return redirect('graphical_method')

    return render(request, "calc/graphical_result.html", {"solution": solution})



def knapsack_method(request):
    if request.method == 'POST':
        # Get basic resources
        total_land = float(request.POST.get('total_land'))
        total_water = float(request.POST.get('total_water'))
        total_labor = float(request.POST.get('total_labor'))
        
        # Get optimization options
        enable_risk = request.POST.get('enable_risk') == 'on'
        enable_market = request.POST.get('enable_market') == 'on'
        
        # Parse crop data with error handling
        crops = []
        i = 0
        while True:
            name = request.POST.get(f'name_{i}')
            if not name:
                break
            try:
                profit = float(request.POST.get(f'profit_{i}'))
                water = float(request.POST.get(f'water_{i}'))
                labor = float(request.POST.get(f'labor_{i}'))
                min_acres = float(request.POST.get(f'min_acres_{i}', 0))
                
                # Apply risk adjustment if enabled
                if enable_risk:
                    profit = apply_risk_adjustment(profit, name)
                
                # Apply market demand adjustment if enabled
                if enable_market:
                    profit = apply_market_adjustment(profit, name)
                
                crops.append({
                    'name': name,
                    'profit': profit,
                    'water': water,
                    'labor': labor,
                    'min_acres': min_acres,
                    'original_profit': profit  # Store original for display
                })
            except (TypeError, ValueError):
                pass  # Skip invalid entries
            i += 1

        # Prepare linear programming problem
        if not crops:
            return render(request, "knapsack_result.html", {
                'error': "No valid crops were provided"
            })

        # Objective: Maximize profit (minimize negative profit)
        c = [-crop['profit'] for crop in crops]
        
        # Inequality constraints (A_ub * x <= b_ub)
        A_ub = [
            [1] * len(crops),  # Land constraint
            [crop['water'] for crop in crops],  # Water constraint
            [crop['labor'] for crop in crops]  # Labor constraint
        ]
        b_ub = [total_land, total_water, total_labor]
        
        # Equality constraints (optional)
        A_eq = []
        b_eq = []
        
        # Bounds (minimum acreage requirements)
        bounds = []
        for crop in crops:
            if crop['min_acres'] > 0:
                bounds.append((crop['min_acres'], None))
            else:
                bounds.append((0, None))
        
        # Solve the linear program
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')
            
            # Process results
            result = []
            total_profit = 0
            total_acres_used = 0
            total_water_used = 0
            total_labor_used = 0
            
            if res.success:
                for i, acres in enumerate(res.x):
                    if acres > 0.001:  # Ignore very small values
                        crop = crops[i]
                        crop_profit = crop['profit'] * acres
                        water_used = crop['water'] * acres
                        labor_used = crop['labor'] * acres
                        
                        result.append({
                            'name': crop['name'],
                            'acres': round(acres, 2),
                            'percent_of_land': round((acres / total_land) * 100, 1),
                            'water_used': round(water_used, 2),
                            'labor_used': round(labor_used, 2),
                            'profit': round(crop_profit, 2),
                            'original_profit_per_acre': crop['original_profit'],
                            'adjusted_profit_per_acre': crop['profit']
                        })
                        
                        total_profit += crop_profit
                        total_acres_used += acres
                        total_water_used += water_used
                        total_labor_used += labor_used
                
                # Calculate utilization percentages
                land_utilization_percent = round((total_acres_used / total_land) * 100, 1)
                water_utilization_percent = round((total_water_used / total_water) * 100, 1)
                labor_utilization_percent = round((total_labor_used / total_labor) * 100, 1)
                
                # Calculate profit per acre
                profit_per_acre = total_profit / total_acres_used if total_acres_used > 0 else 0
                
                # Sensitivity analysis (shadow prices)
                sensitivity_analysis = {
                    'land': calculate_shadow_price(res, 0, 'land', total_land),
                    'water': calculate_shadow_price(res, 1, 'water', total_water),
                    'labor': calculate_shadow_price(res, 2, 'labor', total_labor)
                }
                
                # Generate recommendations
                recommendations = generate_recommendations(
                    result, total_land, total_water, total_labor,
                    land_utilization_percent, water_utilization_percent, labor_utilization_percent,
                    sensitivity_analysis
                )
                
                return render(request, "knapsack_result.html", {
                    'result': result,
                    'total_profit': round(total_profit, 2),
                    'total_acres_used': round(total_acres_used, 2),
                    'total_water_used': round(total_water_used, 2),
                    'total_labor_used': round(total_labor_used, 2),
                    'total_land': total_land,
                    'total_water': total_water,
                    'total_labor': total_labor,
                    'land_utilization_percent': land_utilization_percent,
                    'water_utilization_percent': water_utilization_percent,
                    'labor_utilization_percent': labor_utilization_percent,
                    'profit_per_acre': round(profit_per_acre, 2),
                    'sensitivity_analysis': sensitivity_analysis,
                    'recommendations': recommendations,
                    'enable_risk': enable_risk,
                    'enable_market': enable_market
                })
            else:
                return render(request, "knapsack_result.html", {
                    'error': "No feasible solution found with the given constraints",
                    'total_land': total_land,
                    'total_water': total_water,
                    'total_labor': total_labor
                })
                
        except Exception as e:
            return render(request, "knapsack_result.html", {
                'error': f"An error occurred during optimization: {str(e)}"
            })
    
    return render(request, "knapsack_method.html")

def apply_risk_adjustment(profit, crop_name):
    """Adjust profit based on crop risk factors"""
    risk_factors = {
        'wheat': 0.95,    # 5% reduction for risk
        'rice': 0.90,      # 10% reduction
        'corn': 0.92,
        'soybean': 0.88,
        'cotton': 0.85,
        # Default for other crops
        'default': 0.93
    }
    factor = risk_factors.get(crop_name.lower(), risk_factors['default'])
    return profit * factor

def apply_market_adjustment(profit, crop_name):
    """Adjust profit based on market demand trends"""
    market_factors = {
        'organic': 1.15,   # 15% premium
        'quinoa': 1.20,
        'avocado': 1.10,
        # Default for other crops
        'default': 1.0
    }
    
    # Check if any market term is in the crop name
    for term in market_factors:
        if term != 'default' and term in crop_name.lower():
            return profit * market_factors[term]
    
    return profit * market_factors['default']

def calculate_shadow_price(res, constraint_index, resource_name, total_resource):
    """Calculate shadow price and interpretation for a constraint"""
    if len(res.shadow_prices) > constraint_index:
        shadow_price = -res.shadow_prices[constraint_index]  # Convert to positive since we minimized negative profit
        profit_increase = shadow_price * (total_resource * 0.1)  # Profit increase for 10% more resource
        
        interpretations = {
            'land': "Adding more land would significantly increase profit",
            'water': "Water availability is a key constraint for profit",
            'labor': "Labor availability affects your profit potential"
        }
        
        interpretation = interpretations.get(resource_name, 
            f"Adding more {resource_name} would increase profit by ₹{shadow_price:.2f} per unit")
        
        return {
            'shadow_price': shadow_price,
            'profit_increase': profit_increase,
            'interpretation': interpretation
        }
    return None

def generate_recommendations(result, total_land, total_water, total_labor,
                           land_util_percent, water_util_percent, labor_util_percent,
                           sensitivity_analysis):
    """Generate actionable recommendations based on the solution"""
    recommendations = []
    
    # Land utilization recommendation
    if land_util_percent < 90:
        recommendations.append({
            'title': 'Increase Land Utilization',
            'description': f'Only {land_util_percent}% of your land is being used. Consider adding more crops or increasing acreage.',
            'priority': 'High',
            'badge_color': 'danger'
        })
    else:
        recommendations.append({
            'title': 'Optimal Land Use',
            'description': f'You\'re effectively using {land_util_percent}% of your available land.',
            'priority': 'Low',
            'badge_color': 'success'
        })
    
    # Water utilization recommendation
    if water_util_percent > 95:
        recommendations.append({
            'title': 'Water Constraint',
            'description': f'You\'re using {water_util_percent}% of available water. Consider more water-efficient crops.',
            'priority': 'High',
            'badge_color': 'danger'
        })
    
    # Labor utilization recommendation
    if labor_util_percent < 70:
        recommendations.append({
            'title': 'Underutilized Labor',
            'description': f'Only {labor_util_percent}% of labor is being used. Consider labor-intensive crops.',
            'priority': 'Medium',
            'badge_color': 'warning'
        })
    
    # Shadow price recommendations
    if sensitivity_analysis.get('land', {}).get('shadow_price', 0) > 50:
        recommendations.append({
            'title': 'Expand Land',
            'description': 'Each additional acre could significantly increase profit. Consider leasing more land.',
            'priority': 'High',
            'badge_color': 'primary'
        })
    
    if sensitivity_analysis.get('water', {}).get('shadow_price', 0) > 30:
        recommendations.append({
            'title': 'Improve Water Access',
            'description': 'Water is a limiting resource. Invest in irrigation or water conservation.',
            'priority': 'High',
            'badge_color': 'info'
        })
    
    # Crop-specific recommendations
    if result:
        most_profitable = max(result, key=lambda x: x['profit'])
        least_profitable = min(result, key=lambda x: x['profit'])
        
        recommendations.append({
            'title': f'Focus on {most_profitable["name"]}',
            'description': f'This crop generates ₹{most_profitable["profit"]:.2f} ({most_profitable["percent_of_land"]}% of land). Consider increasing its acreage.',
            'priority': 'Medium',
            'badge_color': 'success'
        })
        
        if least_profitable['profit'] < total_profit * 0.05:  # Less than 5% of total profit
            recommendations.append({
                'title': f'Re-evaluate {least_profitable["name"]}',
                'description': 'This crop contributes minimally to overall profit. Consider replacing it.',
                'priority': 'Medium',
                'badge_color': 'warning'
            })
    
    return recommendations





def simplex_method(request):
    if request.method == 'POST':
        form = FarmerProblemForm(request.POST)
        if form.is_valid():
            problem = form.save(commit=False)
            
            # Get input values
            c = np.array([problem.crop1_profit, problem.crop2_profit])  # Profit coefficients
            A = np.array([
                [1, 1],                     # Land constraint
                [problem.crop1_labor, problem.crop2_labor],  # Labor constraint
                [problem.crop1_water, problem.crop2_water]   # Water constraint
            ])
            b = np.array([problem.total_land, problem.labor_available, problem.water_available])
            
            # Solve using Simplex Method
            solution = solve_simplex(c, A, b)
            
            if solution['status'] == 'optimal':
                problem.crop1_land = solution['x'][0]
                problem.crop2_land = solution['x'][1]
                problem.max_profit = solution['max']
                problem.save()
                return render(request, 'calc/simplex_result.html', {
                    'problem': problem,
                    'solution': solution
                })
            else:
                form.add_error(None, "No feasible solution found with given constraints")
    else:
        form = FarmerProblemForm()
    
    return render(request, 'calc/simplex_method.html', {'form': form})

def solve_simplex(c, A, b):
    """
    Implementation of the Simplex Method for maximization problems
    """
    # Convert to standard form
    num_constraints, num_vars = A.shape
    c = np.array(c).astype(float)
    A = np.array(A).astype(float)
    b = np.array(b).astype(float)
    
    # Add slack variables
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[-1, :num_vars] = -c  # Objective row
    tableau[:-1, :num_vars] = A  # Constraint coefficients
    tableau[:-1, num_vars:num_vars+num_constraints] = np.eye(num_constraints)  # Slack variables
    tableau[:-1, -1] = b  # RHS
    
    # Pivot until optimal solution found
    while True:
        # Find entering variable (most negative in objective row)
        entering_col = np.argmin(tableau[-1, :-1])
        if tableau[-1, entering_col] >= 0:
            break  # Optimal solution found
        
        # Find leaving variable (minimum ratio test)
        ratios = []
        for i in range(num_constraints):
            if tableau[i, entering_col] > 0:
                ratios.append(tableau[i, -1] / tableau[i, entering_col])
            else:
                ratios.append(np.inf)
        leaving_row = np.argmin(ratios)
        
        if ratios[leaving_row] == np.inf:
            return {'status': 'unbounded'}
        
        # Pivot
        pivot_val = tableau[leaving_row, entering_col]
        tableau[leaving_row, :] /= pivot_val
        for i in range(num_constraints + 1):
            if i != leaving_row:
                tableau[i, :] -= tableau[i, entering_col] * tableau[leaving_row, :]
    
    # Extract solution
    solution = np.zeros(num_vars)
    for col in range(num_vars):
        col_data = tableau[:-1, col]
        if np.sum(col_data == 1) == 1 and np.sum(col_data != 0) == 1:
            row = np.where(col_data == 1)[0][0]
            solution[col] = tableau[row, -1]
    
    return {
        'status': 'optimal',
        'x': solution,
        'max': tableau[-1, -1],
        'tableau': tableau
    }

def transportation_method(request):
    return render(request, 'calc/transportation_method.html')

def optimization(request):
    return render(request, 'calc/optimization.html')