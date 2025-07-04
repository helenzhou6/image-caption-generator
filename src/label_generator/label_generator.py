import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import io
import zipfile
import textwrap
import time

# --- Configuration & Data ---

FOOD_CATEGORIES = {
    'Salty Snacks': {
        'products': ["Potato Chips", "Pretzels", "Crackers", "Tortilla Chips", "Popcorn", "Cheese Puffs"],
        'brands': ["Lay's", "Rold Gold", "Ritz", "Doritos", "Cheetos", "Pringles"],
        'base_calories': (140, 220), 'fat_ratio': (0.4, 0.6), 'carb_ratio': (0.3, 0.5), 'protein_ratio': (0.05, 0.1), 'sodium_mg': (150, 450),
        'ingredients': ["Potatoes", "Vegetable Oil (Sunflower, Corn, and/or Canola Oil)", "Salt", "Corn", "Enriched Wheat Flour", "Yeast", "Cheddar Cheese (Milk, Cheese Cultures, Salt, Enzymes)", "Whey", "Buttermilk", "Onion Powder"]
    },
    'Sugary Cereals': {
        'products': ["Frosted Flakes", "Lucky Charms", "Cinnamon Toast Crunch", "Froot Loops", "Cocoa Puffs"],
        'brands': ["Kellogg's", "General Mills", "Post", "Quaker"],
        'base_calories': (110, 190), 'fat_ratio': (0.05, 0.15), 'carb_ratio': (0.7, 0.9), 'protein_ratio': (0.05, 0.1), 'sodium_mg': (100, 250),
        'ingredients': ["Milled Corn", "Sugar", "Malt Flavor", "Whole Grain Oats", "Marshmallows (Sugar, Modified Corn Starch, Corn Syrup, Dextrose)", "Rice Flour", "Canola Oil", "Cocoa Processed with Alkali", "Natural and Artificial Flavor"]
    },
    'Frozen Meals': {
        'products': ["Lasagna", "Chicken Alfredo", "Beef & Broccoli", "Enchiladas", "Mac & Cheese", "Burrito Bowl"],
        'brands': ["Stouffer's", "Lean Cuisine", "Healthy Choice", "Amy's Kitchen", "Marie Callender's"],
        'base_calories': (250, 550), 'fat_ratio': (0.3, 0.5), 'carb_ratio': (0.3, 0.5), 'protein_ratio': (0.2, 0.3), 'sodium_mg': (500, 1200),
        'ingredients': ["Cooked Pasta (Water, Semolina Wheat Flour)", "Tomato Puree (Water, Tomato Paste)", "Cooked Beef", "Low-Moisture Part-Skim Mozzarella Cheese", "Broccoli", "Cooked Rice", "Chicken Breast", "Cream", "Spices"]
    },
    'Yogurt': {
        'products': ["Greek Yogurt", "Low-fat Yogurt", "Fruit on the Bottom", "Vanilla Yogurt", "Strawberry Yogurt"],
        'brands': ["Chobani", "Fage", "Yoplait", "Stonyfield", "Dannon"],
        'base_calories': (80, 180), 'fat_ratio': (0.0, 0.4), 'carb_ratio': (0.2, 0.6), 'protein_ratio': (0.3, 0.5), 'sodium_mg': (50, 150),
        'ingredients': ["Cultured Nonfat Milk", "Water", "Fruit Puree", "Cane Sugar", "Pectin", "Natural Flavors", "Live and Active Cultures", "Vitamin D3", "Lemon Juice Concentrate"]
    },
    'Baked Goods': {
        'products': ["Croissant", "Muffin", "Brownie", "Cookie", "Donut", "Danish"],
        'brands': ["Sara Lee", "Entenmann's", "Little Debbie", "Pepperidge Farm", "Hostess"],
        'base_calories': (250, 500), 'fat_ratio': (0.4, 0.6), 'carb_ratio': (0.4, 0.6), 'protein_ratio': (0.05, 0.15), 'sodium_mg': (150, 450),
        'ingredients': ["Enriched Bleached Flour", "Sugar", "Vegetable Shortening (Palm Oil, Soybean Oil)", "Cocoa Processed with Alkali", "Eggs", "Butter (Cream, Salt)", "Vanilla Extract", "Leavening (Baking Soda)", "Soy Lecithin"]
    },
    'Beverages': {
        'products': ["Orange Juice", "Apple Juice", "Energy Drink", "Sports Drink", "Soda"],
        'brands': ["Tropicana", "Minute Maid", "Red Bull", "Gatorade", "Coca-Cola"],
        'base_calories': (80, 250), 'fat_ratio': (0.0, 0.05), 'carb_ratio': (0.9, 1.0), 'protein_ratio': (0.0, 0.05), 'sodium_mg': (10, 200),
        'ingredients': ["Carbonated Water", "High Fructose Corn Syrup", "Natural Flavors", "Citric Acid", "Orange Juice Concentrate", "Sodium Benzoate (to protect taste)", "Caffeine", "Phosphoric Acid"]
    }
}

SERVING_SIZES = [
    "1 cup (228g)", "2/3 cup (55g)", "1 package (255g)", "100g", "30g", "1 slice (28g)",
    "1 container (150g)", "1 piece (45g)", "2 pieces (60g)", "1 bar (40g)", "12 fl oz (355ml)",
    "8 fl oz (240ml)", "1 bottle (500ml)", "17 chips (28g)", "3 pretzels (28g)", "1 serving (85g)"
]

# --- TEMPLATE CONFIGURATION (with new title style properties) ---
TEMPLATES = [
    {'name': 'Classic FDA (US)', 'style': 'standard', 'bg_color': '#FFFFFF', 'line_color': '#000000', 'dual_column': False, 'title_size_mult': 1.2, 'title_weight': 'bold'},
    {'name': 'US Dual Column', 'style': 'us_dual_column', 'bg_color': '#FFFFFF', 'line_color': '#000000', 'dual_column': True, 'title_size_mult': 1.0, 'title_weight': 'bold'},
    {'name': 'Clean Modern (US)', 'style': 'standard', 'bg_color': '#FAFAFA', 'line_color': '#333333', 'dual_column': False, 'title_size_mult': 1.1, 'title_weight': 'regular'},
    {'name': 'Basic UK/EU', 'style': 'uk_style', 'bg_color': '#FFFFFF', 'line_color': '#000000', 'title_size_mult': 1.0, 'title_weight': 'bold'},
    {'name': 'UK Minimal', 'style': 'uk_style', 'bg_color': '#FFFFFF', 'line_color': '#CCCCCC', 'title_size_mult': 0.9, 'title_weight': 'regular'},
    # This template will have the boxed, reverse-contrast title
    {'name': 'US Compact', 'style': 'standard', 'bg_color': '#F8F8F8', 'line_color': '#666666', 'dual_column': False, 'title_size_mult': 1.0, 'title_weight': 'bold', 'title_style': 'boxed'},
    {'name': 'UK Clean', 'style': 'uk_style', 'bg_color': '#FAFAFA', 'line_color': '#999999', 'title_size_mult': 1.0, 'title_weight': 'bold'}
]

# --- Data Generation Logic ---

def generate_label_data():
    """Generates a dictionary of random nutrition data."""
    cat_name = random.choice(list(FOOD_CATEGORIES.keys()))
    category = FOOD_CATEGORIES[cat_name]

    servings_val = random.choice([1, 2, 4, 'about 2', 'about 4', 'about 8'])
    possible_templates = TEMPLATES
    if servings_val == 1:
        possible_templates = [t for t in TEMPLATES if not t.get('dual_column', False)]

    # Make a copy of the template to allow for modification
    template = random.choice(possible_templates).copy()

    # --- Random Background Color Logic (80/20 split) ---
    if random.random() < 0.2:
        colors = ['#FFFFE0', '#FFDAB9', '#90EE90', '#F5DEB3', '#FFEBCD']  # Light Yellows, Oranges, Greens
        template['bg_color'] = random.choice(colors)


    calories = random.randint(*category['base_calories'])
    fat_cals = calories * random.uniform(*category['fat_ratio'])
    carb_cals = calories * random.uniform(*category['carb_ratio'])
    prot_cals = max(0, calories - fat_cals - carb_cals)
    fat = round(fat_cals / 9, 1)
    carbs = round(carb_cals / 4)
    protein = round(prot_cals / 4)
    sodium = random.randint(*category['sodium_mg'])
    salt = round(sodium / 1000 * 2.5, 2)
    sat_fat = round(fat * random.uniform(0.1, 0.4), 1)
    trans_fat = 0 if random.random() > 0.3 else round(fat * random.uniform(0.01, 0.05), 1)
    poly_fat = round(fat * random.uniform(0.1, 0.3), 1)
    mono_fat = max(0, round(fat - sat_fat - trans_fat - poly_fat, 1))
    cholesterol = random.randint(0, 100) if random.random() > 0.4 else 0
    fiber = round(carbs * random.uniform(0.05, 0.15))
    sugars = round(carbs * random.uniform(0.1, 0.7))
    added_sugars = round(sugars * random.uniform(0.5, 1.0)) if random.random() > 0.5 else 0
    vitamin_d = random.randint(0, 15) if random.random() > 0.5 else 0
    calcium = random.randint(10, 500) if random.random() > 0.4 else 0
    iron = round(random.uniform(0.5, 4.0), 1) if random.random() > 0.4 else 0
    potassium = random.randint(100, 800) if random.random() > 0.5 else 0
    num_ingredients = random.randint(4, min(10, len(category['ingredients'])))
    ingredients_text = ", ".join(random.sample(category['ingredients'], num_ingredients)) + "."

    return {
        'servingSize': random.choice(SERVING_SIZES), 'servingsPerContainer': servings_val,
        'calories': int(calories), 'fat': fat, 'saturatedFat': sat_fat, 'transFat': trans_fat,
        'polyunsaturatedFat': poly_fat, 'monounsaturatedFat': mono_fat,
        'cholesterol': cholesterol, 'sodium': sodium, 'salt': salt,
        'carbs': carbs, 'fiber': fiber, 'sugars': sugars, 'addedSugars': added_sugars,
        'protein': protein, 'vitaminD': vitamin_d, 'calcium': calcium, 'iron': iron,
        'potassium': potassium, 'template': template, 'ingredients': ingredients_text
    }

# --- Image Drawing Engine ---

def _draw_standard_label(draw, data, width, height, fonts, p, line_color):
    """Draws a standard single-column US FDA-style label and returns the final y-coordinate."""
    y = p
    line_width, medium_line_width, thick_line_width = max(1, int(height/200)), max(2, int(height/100)), max(4, int(height/60))

    # --- Title Drawing Logic ---
    title_text = "Nutrition Facts"
    template = data['template']
    if template.get('title_style') == 'boxed':
        title_box_h = fonts['h1'].size * 1.8
        draw.rectangle([(p, y), (width - p, y + title_box_h)], fill=template.get('line_color', '#000000'))
        draw.text((p + (width-2*p)/2, y + title_box_h/2), title_text, font=fonts['h1'], fill=template.get('bg_color', '#FFFFFF'), anchor="mm")
        y += title_box_h + int(height * 0.02)
    else:
        draw.text((p, y), title_text, font=fonts['h1'], fill="black")
        y += int(fonts['h1'].size * 1.5)

    servings_text = f"{data['servingsPerContainer']} servings per container" if data['servingsPerContainer'] != 1 else "1 serving per container"
    draw.text((p, y), servings_text, font=fonts['body_r'], fill="black")
    y += int(height * 0.04)
    draw.text((p, y), "Serving size", font=fonts['body_b'], fill="black")
    draw.text((width - p, y), data['servingSize'], font=fonts['body_b'], fill="black", anchor="ra")
    y += int(height * 0.04)

    draw.line([(p, y), (width - p, y)], fill=line_color, width=thick_line_width)
    y += int(height * 0.025)
    draw.text((p, y), "Amount per serving", font=fonts['body_r'], fill="black")
    y += int(height * 0.04)
    draw.text((p, y), "Calories", font=fonts['h1'], fill="black")
    draw.text((width - p, y), str(data['calories']), font=fonts['h1'], fill="black", anchor="ra")
    y += int(height * 0.06)

    draw.line([(p, y), (width - p, y)], fill=line_color, width=medium_line_width)
    y += int(height * 0.02)
    draw.text((width - p, y), "% Daily Value*", font=fonts['body_b'], fill="black", anchor="ra")
    y += int(height * 0.04)

    def draw_nutrient(name, amount_str, dv, indent=0, bold=True, line=True):
        nonlocal y
        line_h = int(height * 0.04)
        font = fonts['body_b'] if bold else fonts['body_r']
        indent_px = p + (p * indent)
        draw.text((indent_px, y), name, font=font, fill="black")
        dv_text = f"{dv}%" if dv != "" else ""
        dv_width = draw.textlength(dv_text, font=fonts['body_b']) if dv_text else 0
        spacing = int(width * 0.04)
        if dv_text: draw.text((width - p, y), dv_text, font=fonts['body_b'], fill="black", anchor="ra")
        if amount_str:
            amount_x = width - p - dv_width - spacing
            draw.text((amount_x, y), amount_str, font=font, fill="black", anchor="ra")
        y += line_h
        if line: draw.line([(indent_px, y - line_h/3), (width - p, y - line_h/3)], fill=line_color, width=line_width)

    draw_nutrient("Total Fat", f"{data['fat']}g", min(100, round(data['fat'] / 78 * 100)))
    draw_nutrient("Saturated Fat", f"{data['saturatedFat']}g", min(100, round(data['saturatedFat'] / 20 * 100)), indent=1, bold=False)
    draw_nutrient("Trans Fat", f"{data['transFat']}g", "", indent=1, bold=False)
    draw_nutrient("Cholesterol", f"{data['cholesterol']}mg", min(100, round(data['cholesterol'] / 300 * 100)))
    draw_nutrient("Sodium", f"{data['sodium']}mg", min(100, round(data['sodium'] / 2300 * 100)))
    draw_nutrient("Total Carbohydrate", f"{data['carbs']}g", min(100, round(data['carbs'] / 275 * 100)))
    draw_nutrient("Dietary Fiber", f"{data['fiber']}g", min(100, round(data['fiber'] / 28 * 100)), indent=1, bold=False)
    draw_nutrient("Total Sugars", f"{data['sugars']}g", "", indent=1, bold=False)
    if data['addedSugars'] > 0:
        draw_nutrient(f"Includes {data['addedSugars']}g Added Sugars", "", min(100, round(data['addedSugars'] / 50 * 100)), indent=2, bold=False)
    draw_nutrient("Protein", f"{data['protein']}g", min(100, round(data['protein'] / 50 * 100)), line=False)

    y += int(height * 0.025)
    draw.line([(p, y), (width - p, y)], fill=line_color, width=thick_line_width)
    y += int(height * 0.02)
    if data['vitaminD'] > 0: draw_nutrient("Vitamin D", f"{data['vitaminD']}mcg", min(100, round(data['vitaminD'] / 20 * 100)), bold=False)
    if data['calcium'] > 0: draw_nutrient("Calcium", f"{data['calcium']}mg", min(100, round(data['calcium'] / 1300 * 100)), bold=False)
    if data['iron'] > 0: draw_nutrient("Iron", f"{data['iron']}mg", min(100, round(data['iron'] / 18 * 100)), bold=False)
    if data['potassium'] > 0: draw_nutrient("Potassium", f"{data['potassium']}mg", min(100, round(data['potassium'] / 4700 * 100)), bold=False, line=False)

    y += int(height * 0.03)
    draw.line([(p, y), (width - p, y)], fill=line_color, width=medium_line_width)
    y += int(height * 0.02)
    disclaimer = "* The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a daily diet. 2,000 calories a day is used for general nutrition advice."
    wrapped_disclaimer = textwrap.fill(disclaimer, width=int(width / (fonts['small'].size * 0.6)))
    for line in wrapped_disclaimer.split('\n'):
        draw.text((p, y), line, font=fonts['small'], fill="black")
        y += int(height * 0.025)
    return y

def _draw_uk_style_label(draw, data, width, height, fonts, p, line_color):
    """Draws a tabular UK/EU-style label and returns the final y-coordinate."""
    y = p
    line_h = int(height * 0.045)
    template = data['template']

    # --- Title Drawing Logic ---
    title_text = "Nutrition Information"
    if template.get('title_style') == 'boxed':
        title_box_h = fonts['h1'].size * 1.8
        draw.rectangle([(p, y), (width - p, y + title_box_h)], fill=template.get('line_color', '#000000'))
        draw.text((p + (width - 2 * p) / 2, y + title_box_h / 2), title_text, font=fonts['h1'], fill=template.get('bg_color', '#FFFFFF'), anchor="mm")
        y += title_box_h + int(height * 0.02)
    else:
        draw.text((p, y), title_text, font=fonts['h1'], fill="black")
        y += int(fonts['h1'].size * 1.5)

    col2_center, col3_center = width * 0.55, width * 0.82
    serving_header_text = f"Per serving ({data['servingSize']})"
    col3_width_pixels = (width - p) - (col2_center + (width - col3_center))
    avg_char_width = fonts['body_b'].size * 0.5
    wrap_width = max(15, int(col3_width_pixels / avg_char_width))
    wrapped_serving_header = textwrap.fill(serving_header_text, width=wrap_width)
    header_lines = wrapped_serving_header.split('\n')
    header_y_start = y
    draw.text((col2_center, header_y_start), "Per 100g", font=fonts['body_b'], fill="black", anchor="ma")
    temp_y = header_y_start
    for line in header_lines:
        draw.text((col3_center, temp_y), line, font=fonts['body_b'], fill="black", anchor="ma")
        temp_y += int(fonts['body_b'].size * 1.2)
    y = temp_y + int(line_h * 0.2)
    draw.line([(p, y), (width - p, y)], fill=line_color, width=2)
    y += int(line_h * 0.5)

    def draw_uk_nutrient(name, val_100g, val_portion, ri_portion, indent=0):
        nonlocal y
        indent_px = p + (p * indent)
        draw.text((indent_px, y), name, font=fonts['body_r'], fill="black")
        draw.text((col2_center, y), val_100g, font=fonts['body_r'], fill="black", anchor="ma")
        ri_text = f" ({ri_portion}%)" if ri_portion != "-" else ""
        draw.text((col3_center, y), f"{val_portion}{ri_text}", font=fonts['body_r'], fill="black", anchor="ma")
        y += line_h
        draw.line([(indent_px, y - line_h/3), (width - p, y - line_h/3)], fill=line_color, width=1)

    energy_kj = int(data['calories'] * 4.184)
    draw.text((p, y), "Energy", font=fonts['body_b'], fill="black")
    energy_text = f"{energy_kj}kJ / {data['calories']}kcal"
    draw.text((col2_center, y), energy_text, font=fonts['body_r'], fill="black", anchor="ma")
    draw.text((col3_center, y), energy_text, font=fonts['body_r'], fill="black", anchor="ma")
    y += line_h
    draw.line([(p, y - line_h/3), (width - p, y - line_h/3)], fill=line_color, width=1)
    draw_uk_nutrient("Fat", f"{data['fat']:.1f}g", f"{data['fat']:.1f}g", min(100, round(data['fat'] / 70 * 100)))
    draw_uk_nutrient(" of which saturates", f"{data['saturatedFat']:.1f}g", f"{data['saturatedFat']:.1f}g", min(100, round(data['saturatedFat'] / 20 * 100)), indent=1)
    draw_uk_nutrient("Carbohydrate", f"{data['carbs']:.0f}g", f"{data['carbs']:.0f}g", min(100, round(data['carbs'] / 260 * 100)))
    draw_uk_nutrient(" of which sugars", f"{data['sugars']:.0f}g", f"{data['sugars']:.0f}g", min(100, round(data['sugars'] / 90 * 100)), indent=1)
    draw_uk_nutrient("Fibre", f"{data['fiber']:.0f}g", f"{data['fiber']:.0f}g", "-")
    draw_uk_nutrient("Protein", f"{data['protein']:.0f}g", f"{data['protein']:.0f}g", min(100, round(data['protein'] / 50 * 100)))
    draw_uk_nutrient("Salt", f"{data['salt']:.2f}g", f"{data['salt']:.2f}g", min(100, round(data['salt'] / 6 * 100)))
    y += int(height * 0.02)
    disclaimer = "Reference intake of an average adult (8400kJ/2000kcal)"
    wrapped_disclaimer = textwrap.fill(disclaimer, width=int(width / (fonts['small'].size * 0.6)))
    for line in wrapped_disclaimer.split('\n'):
        draw.text((p, y), line, font=fonts['small'], fill="black")
        y += int(height * 0.025)
    return y

def _draw_dual_column_label(draw, data, width, height, fonts, p, line_color):
    """Draws a dual-column US FDA-style label and returns the final y-coordinate."""
    y = p
    line_width, medium_line_width, thick_line_width = max(1, int(height/200)), max(2, int(height/100)), max(4, int(height/60))
    template = data['template']

    # --- Title Drawing Logic ---
    title_text = "Nutrition Facts"
    if template.get('title_style') == 'boxed':
        title_box_h = fonts['h1'].size * 1.8
        draw.rectangle([(p, y), (width - p, y + title_box_h)], fill=template.get('line_color', '#000000'))
        draw.text((p + (width-2*p)/2, y + title_box_h/2), title_text, font=fonts['h1'], fill=template.get('bg_color', '#FFFFFF'), anchor="mm")
        y += title_box_h + int(height * 0.02)
    else:
        draw.text((p, y), title_text, font=fonts['h1'], fill="black")
        y += int(fonts['h1'].size * 1.5)

    servings = data['servingsPerContainer']
    servings_text = f"{servings} servings per container"
    draw.text((p, y), servings_text, font=fonts['body_r'], fill="black")
    y += int(height * 0.04)
    draw.text((p, y), "Serving size", font=fonts['body_b'], fill="black")
    draw.text((width - p, y), data['servingSize'], font=fonts['body_b'], fill="black", anchor="ra")
    y += int(height * 0.04)
    draw.line([(p, y), (width - p, y)], fill=line_color, width=medium_line_width)
    y += int(height * 0.02)
    col1_x, col2_x = width * 0.6, width - p
    draw.text((col1_x, y), "Per serving", font=fonts['body_b'], fill="black", anchor="ma")
    draw.text((col2_x, y), "Per container", font=fonts['body_b'], fill="black", anchor="ra")
    y += int(height * 0.04)
    container_multiplier = int(str(servings).replace('about ', ''))
    draw.text((p, y), "Calories", font=fonts['h1'], fill="black")
    draw.text((col1_x, y), str(data['calories']), font=fonts['h1'], fill="black", anchor="ma")
    draw.text((col2_x, y), str(data['calories'] * container_multiplier), font=fonts['h1'], fill="black", anchor="ra")
    y += int(height * 0.06)
    draw.line([(p, y), (width - p, y)], fill=line_color, width=thick_line_width)
    y += int(height * 0.02)
    draw.text((col2_x, y), "% Daily Value*", font=fonts['body_b'], fill="black", anchor="ra")
    y += int(height * 0.04)

    def draw_dual_nutrient(name, val_serving, val_container, dv_container, indent=0, bold=True, unit='g'):
        nonlocal y
        line_h = int(height * 0.04)
        font = fonts['body_b'] if bold else fonts['body_r']
        indent_px = p + (p * indent)
        spacing = int(width * 0.04)
        draw.text((indent_px, y), name, font=font, fill="black")
        draw.text((col1_x, y), f"{val_serving}{unit}", font=font, fill="black", anchor="ma")
        container_text = f"{val_container}{unit}"
        dv_text = f"{dv_container}%" if dv_container != "" else ""
        if dv_text:
            draw.text((col2_x, y), dv_text, font=fonts['body_b'], fill="black", anchor="ra")
            dv_width = draw.textlength(dv_text, font=fonts['body_b'])
            draw.text((col2_x - dv_width - spacing, y), container_text, font=font, fill="black", anchor="ra")
        else: draw.text((col2_x, y), container_text, font=font, fill="black", anchor="ra")
        y += line_h
        draw.line([(indent_px, y - line_h/3), (width - p, y - line_h/3)], fill=line_color, width=line_width)

    draw_dual_nutrient("Total Fat", data['fat'], round(data['fat']*container_multiplier,1), min(100, round(data['fat']*container_multiplier/78*100)))
    draw_dual_nutrient("Saturated Fat", data['saturatedFat'], round(data['saturatedFat']*container_multiplier,1), min(100, round(data['saturatedFat']*container_multiplier/20*100)), indent=1, bold=False)
    draw_dual_nutrient("Sodium", data['sodium'], round(data['sodium']*container_multiplier), min(100, round(data['sodium']*container_multiplier/2300*100)), unit='mg')
    draw_dual_nutrient("Total Carbohydrate", data['carbs'], round(data['carbs']*container_multiplier), min(100, round(data['carbs']*container_multiplier/275*100)))
    draw_dual_nutrient("Protein", data['protein'], round(data['protein']*container_multiplier), min(100, round(data['protein']*container_multiplier/50*100)))
    y += int(height * 0.03)
    disclaimer = "* The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a daily diet. 2,000 calories a day is used for general nutrition advice."
    wrapped_disclaimer = textwrap.fill(disclaimer, width=int(width / (fonts['small'].size * 0.6)))
    for line in wrapped_disclaimer.split('\n'):
        draw.text((p, y), line, font=fonts['small'], fill="black")
        y += int(height * 0.025)
    return y

def draw_label(data, width, height):
    """Main drawing dispatcher."""
    template = data['template']
    img = Image.new('RGB', (width, height), template['bg_color'])
    draw = ImageDraw.Draw(img)

    try:
        # --- Create fonts based on template properties ---
        title_size_mult = template.get('title_size_mult', 1.0)
        title_weight = template.get('title_weight', 'bold')
        title_font_file = "DejaVuSans-Bold.ttf" if title_weight == 'bold' else "DejaVuSans.ttf"

        font_size_h1 = int((height / 22) * title_size_mult)
        font_size_body = int(height / 40)
        font_size_small = int(height / 55)
        fonts = {
            'h1': ImageFont.truetype(title_font_file, font_size_h1),
            'body_b': ImageFont.truetype("DejaVuSans-Bold.ttf", font_size_body),
            'body_r': ImageFont.truetype("DejaVuSans.ttf", font_size_body),
            'small': ImageFont.truetype("DejaVuSans.ttf", font_size_small)
        }
    except IOError:
        st.warning("DejaVu fonts not found, using default font. Layout may be affected.", icon="⚠️")
        font_size_body, font_size_small = 12, 10
        fonts = {'h1': ImageFont.load_default(), 'body_b': ImageFont.load_default(), 'body_r': ImageFont.load_default(), 'small': ImageFont.load_default()}
        if not hasattr(fonts['h1'], 'size'): fonts['h1'].size = 15
        if not hasattr(fonts['body_b'], 'size'): fonts['body_b'].size = font_size_body
        if not hasattr(fonts['small'], 'size'): fonts['small'].size = font_size_small

    p = int(width * 0.05)
    line_color = template.get('line_color', '#000000')

    if template['style'] == 'us_dual_column':
        final_y = _draw_dual_column_label(draw, data, width, height, fonts, p, line_color)
    elif template['style'] == 'uk_style':
        final_y = _draw_uk_style_label(draw, data, width, height, fonts, p, line_color)
    else:
        final_y = _draw_standard_label(draw, data, width, height, fonts, p, line_color)

    y_ingredients = final_y + int(height * 0.04)
    ingredients_header_h, line_h = int(height * 0.04), int(height * 0.025)
    if y_ingredients < height - (ingredients_header_h + 3 * line_h):
        draw.line([(p, y_ingredients), (width - p, y_ingredients)], fill=line_color, width=max(1, int(height/200)))
        y_ingredients += int(height * 0.02)
        draw.text((p, y_ingredients), "INGREDIENTS:", font=fonts['body_b'], fill="black")
        y_ingredients += ingredients_header_h
        avg_char_width = fonts['small'].size * 0.6
        wrap_width = int((width - 2 * p) / avg_char_width)
        ingredients_wrapped = textwrap.fill(data['ingredients'], width=wrap_width)
        for line in ingredients_wrapped.split('\n'):
            if y_ingredients > height - (line_h + p/2): break
            draw.text((p, y_ingredients), line, font=fonts['small'], fill="black")
            y_ingredients += line_h
    return img

# --- Streamlit Interface ---

def main():
    st.set_page_config(page_title="Nutrition Label Generator", layout="wide")
    st.title("️Nutrition Label Generator")
    st.markdown("Generate batches of synthetic nutrition labels. All categories and templates are randomly distributed.")

    if 'running' not in st.session_state: st.session_state.running = False
    if 'batch_number' not in st.session_state: st.session_state.batch_number = 0
    if 'previews' not in st.session_state: st.session_state.previews = []

    st.sidebar.header("Generation Settings")
    st.sidebar.subheader("Image Dimensions")
    width = st.sidebar.slider("Width (px)", 400, 1000, 600, key="img_width")
    height = st.sidebar.slider("Height (px)", 600, 1400, 900, key="img_height")

    st.sidebar.subheader("Actions")
    if st.sidebar.button("Show Preview (10 Images)"):
        st.session_state.running = False
        preview_images = []
        for _ in range(10):
            data = generate_label_data()
            img = draw_label(data, width, height)
            if img: preview_images.append(img)
        st.session_state.previews = preview_images

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🟢 Start Producing"):
            st.session_state.previews = []
            if not st.session_state.running:
                st.session_state.running = True
                st.session_state.batch_number = 1
                st.rerun()
    with col2:
        if st.button("🔴 Stop"):
            if st.session_state.running:
                st.session_state.running = False
                st.rerun()

    if st.session_state.previews:
        st.subheader("🖼️ Preview")
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.previews):
            with cols[i % 5]:
                st.image(img, caption=f"Preview {i+1}", use_container_width=True)
        st.info("Previews are shown above. Click 'Start Producing' to generate downloadable batches.")

    if st.session_state.running:
        st.subheader(f"⚙️ Processing Batch #{st.session_state.batch_number}")
        BATCH_SIZE = 1000
        images_for_zip = []
        start_time = time.time()
        progress_bar = st.progress(0, text=f"Generating {BATCH_SIZE} labels...")
        for i in range(BATCH_SIZE):
            data = generate_label_data()
            img = draw_label(data, st.session_state.img_width, st.session_state.img_height)
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG', dpi=(300, 300))
                img_bytes.seek(0)
                images_for_zip.append({'filename': f"label_batch{st.session_state.batch_number}_{i+1:04d}.png", 'bytes': img_bytes.getvalue()})
            progress_bar.progress((i + 1) / BATCH_SIZE, text=f"Generating label {i+1}/{BATCH_SIZE}...")
        end_time = time.time()
        elapsed = end_time - start_time
        st.success(f"Batch #{st.session_state.batch_number} generated in {elapsed:.2f} seconds.")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in images_for_zip:
                zip_file.writestr(item['filename'], item['bytes'])
        zip_buffer.seek(0)
        st.download_button(label=f"📦 Download Batch #{st.session_state.batch_number} ({BATCH_SIZE} images)", data=zip_buffer, file_name=f"nutrition_labels_batch_{st.session_state.batch_number}.zip", mime="application/zip", key=f"batch_dl_{st.session_state.batch_number}")
        st.session_state.batch_number += 1
        st.rerun()
    elif not st.session_state.previews:
        st.info("Select an action from the sidebar to begin.")

if __name__ == "__main__":
    main()