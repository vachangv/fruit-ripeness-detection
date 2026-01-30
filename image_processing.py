import cv2
import numpy as np

def analyze_image(image_path, fruit_type="banana"):
    """
    Main analysis function
    Returns: color, ripeness, quality, confidence, recommendation
    """
    image = cv2.imread(image_path)

    if image is None:
        return "Invalid Image", "Unknown", "Unknown", 0, "Please upload a valid image"

    # Crop center region to reduce background effect
    h, w, _ = image.shape
    image = image[int(0.1*h):int(0.9*h), int(0.1*w):int(0.9*w)]

    # Resize for consistent processing
    image = cv2.resize(image, (400, 400))
    
    # Apply slight blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Remove background
    fruit_only = remove_background(image)
    
    hsv = cv2.cvtColor(fruit_only, cv2.COLOR_BGR2HSV)
    
    # Get fruit-specific features based on type
    if fruit_type.lower() == "banana":
        features = extract_banana_features(hsv, fruit_only)
        ripeness, confidence = detect_banana_ripeness(features)
        color = get_banana_color(features)
        quality = detect_banana_quality(features)
    
    elif fruit_type.lower() == "apple":
        features = extract_apple_features(hsv, fruit_only)
        ripeness, confidence = detect_apple_ripeness(features)
        color = get_apple_color(features)
        quality = detect_apple_quality(features)
    
    elif fruit_type.lower() == "mango":
        features = extract_mango_features(hsv, fruit_only)
        ripeness, confidence = detect_mango_ripeness(features)
        color = get_mango_color(features)
        quality = detect_mango_quality(features)
    
    elif fruit_type.lower() == "orange":
        features = extract_orange_features(hsv, fruit_only)
        ripeness, confidence = detect_orange_ripeness(features)
        color = get_orange_color(features)
        quality = detect_orange_quality(features)
    
    elif fruit_type.lower() == "tomato":
        features = extract_tomato_features(hsv, fruit_only)
        ripeness, confidence = detect_tomato_ripeness(features)
        color = get_tomato_color(features)
        quality = detect_tomato_quality(features)
        
        # added extra fruits 
    
    elif fruit_type.lower() == "strawberry":
        features = extract_strawberry_features(hsv, fruit_only)
        ripeness, confidence = detect_strawberry_ripeness(features)
        color = get_strawberry_color(features)
        quality = detect_strawberry_quality(features)
    
    elif fruit_type.lower() == "grapes":
        features = extract_grapes_features(hsv, fruit_only)
        ripeness, confidence = detect_grapes_ripeness(features)
        color = get_grapes_color(features)
        quality = detect_grapes_quality(features)
    
    elif fruit_type.lower() == "avocado":
        features = extract_avocado_features(hsv, fruit_only)
        ripeness, confidence = detect_avocado_ripeness(features)
        color = get_avocado_color(features)
        quality = detect_avocado_quality(features)
    
    elif fruit_type.lower() == "pineapple":
        features = extract_pineapple_features(hsv, fruit_only)
        ripeness, confidence = detect_pineapple_ripeness(features)
        color = get_pineapple_color(features)
        quality = detect_pineapple_quality(features)
    
    elif fruit_type.lower() == "papaya":
        features = extract_papaya_features(hsv, fruit_only)
        ripeness, confidence = detect_papaya_ripeness(features)
        color = get_papaya_color(features)
        quality = detect_papaya_quality(features)
    
    else:
        return "Unknown Fruit", "Unknown", "Unknown", 0, "Fruit type not supported"
    
    # Generate AI recommendation
    recommendation = generate_recommendation(fruit_type, ripeness, quality, confidence)

    return color, ripeness, quality, confidence, recommendation


# ---------- BACKGROUND REMOVAL ----------
def remove_background(image):
    """Remove background using GrabCut algorithm"""
    
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    h, w = image.shape[:2]
    rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    result[mask2 == 0] = [255, 255, 255]
    
    return result


# ========== BANANA DETECTION ==========
def extract_banana_features(hsv, bgr_image):
    """Extract color features from banana"""
    h, s, v = cv2.split(hsv)
    
    banana_mask = (s > 20) & (v > 30) & (v < 250) & ~((h == 0) & (s < 10) & (v > 200))
    
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    
    yellow_lower = np.array([20, 80, 100])
    yellow_upper = np.array([35, 255, 255])
    
    brown_lower = np.array([8, 60, 20])
    brown_upper = np.array([20, 255, 200])
    
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & banana_mask
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) & banana_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & banana_mask
    black_mask = cv2.inRange(hsv, black_lower, black_upper) & banana_mask
    
    total_valid = np.count_nonzero(banana_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'yellow_ratio': np.count_nonzero(yellow_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
        'black_ratio': np.count_nonzero(black_mask) / max(total_valid, 1),
        'mean_hue': np.mean(h[banana_mask]) if total_valid > 0 else 0,
    }
    
    return features

def detect_banana_ripeness(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    spot_ratio = features['brown_ratio'] + features['black_ratio']
    
    if green_ratio > 0.4 or (green_ratio > 0.25 and yellow_ratio < 0.3):
        confidence = min(95, 70 + green_ratio * 50)
        return "Unripe", int(confidence)
    
    elif spot_ratio > 0.15 or features['black_ratio'] > 0.05:
        confidence = min(95, 60 + spot_ratio * 100)
        return "Overripe", int(confidence)
    
    elif yellow_ratio > 0.3 and spot_ratio < 0.15:
        if green_ratio > 0.15:
            return "Ripe (Early)", 75
        else:
            confidence = min(95, 70 + yellow_ratio * 40)
            return "Ripe", int(confidence)
    
    elif yellow_ratio > 0.15 and green_ratio > 0.15:
        return "Turning Ripe", 70
    
    else:
        if features['mean_hue'] > 50:
            return "Unripe", 60
        elif features['mean_hue'] > 22:
            return "Ripe", 65
        else:
            return "Overripe", 60

def get_banana_color(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    spot_ratio = features['brown_ratio'] + features['black_ratio']
    
    if green_ratio > yellow_ratio and green_ratio > 0.25:
        return "Green"
    elif spot_ratio > 0.2:
        return "Brown-Spotted"
    elif spot_ratio > 0.1:
        return "Yellow with Spots"
    elif yellow_ratio > 0.3:
        return "Yellow"
    else:
        return "Yellow-Green"

def detect_banana_quality(features):
    spot_ratio = features['brown_ratio'] + features['black_ratio']
    yellow_ratio = features['yellow_ratio']
    green_ratio = features['green_ratio']
    
    if green_ratio > 0.3:
        return "Grade B (Unripe)" if spot_ratio < 0.05 else "Grade C (Unripe with defects)"
    elif yellow_ratio > 0.4 and spot_ratio < 0.05:
        return "Grade A (Premium)"
    elif yellow_ratio > 0.25 and spot_ratio < 0.12:
        return "Grade B (Good)"
    elif spot_ratio > 0.15:
        return "Grade C (Overripe)"
    else:
        return "Grade B (Acceptable)" if spot_ratio < 0.08 else "Grade C (Fair)"


# ========== APPLE DETECTION ==========
def extract_apple_features(hsv, bgr_image):
    """Extract color features from apple"""
    h, s, v = cv2.split(hsv)
    
    apple_mask = (s > 15) & (v > 30) & (v < 250)
    
    green_lower = np.array([35, 30, 30])
    green_upper = np.array([85, 255, 255])
    
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    
    yellow_lower = np.array([20, 50, 100])
    yellow_upper = np.array([35, 255, 255])
    
    brown_lower = np.array([8, 40, 20])
    brown_upper = np.array([20, 255, 150])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & apple_mask
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1) & apple_mask
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2) & apple_mask
    red_mask = red_mask1 | red_mask2
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) & apple_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & apple_mask
    
    total_valid = np.count_nonzero(apple_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'red_ratio': np.count_nonzero(red_mask) / max(total_valid, 1),
        'yellow_ratio': np.count_nonzero(yellow_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_apple_ripeness(features):
    green_ratio = features['green_ratio']
    red_ratio = features['red_ratio']
    brown_ratio = features['brown_ratio']
    
    if green_ratio > 0.5:
        confidence = min(90, 65 + green_ratio * 40)
        return "Unripe", int(confidence)
    
    elif red_ratio > 0.4 and brown_ratio < 0.1:
        confidence = min(95, 70 + red_ratio * 50)
        return "Ripe", int(confidence)
    
    elif brown_ratio > 0.15:
        confidence = min(90, 60 + brown_ratio * 80)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 70

def get_apple_color(features):
    green_ratio = features['green_ratio']
    red_ratio = features['red_ratio']
    
    if red_ratio > 0.4:
        return "Red"
    elif green_ratio > 0.4:
        return "Green"
    else:
        return "Red-Green Mix"

def detect_apple_quality(features):
    brown_ratio = features['brown_ratio']
    red_ratio = features['red_ratio']
    
    if red_ratio > 0.5 and brown_ratio < 0.05:
        return "Grade A (Premium)"
    elif brown_ratio < 0.12:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== MANGO DETECTION ==========
def extract_mango_features(hsv, bgr_image):
    """Extract color features from mango"""
    h, s, v = cv2.split(hsv)
    
    mango_mask = (s > 20) & (v > 30) & (v < 250)
    
    green_lower = np.array([35, 30, 30])
    green_upper = np.array([85, 255, 255])
    
    yellow_lower = np.array([20, 60, 100])
    yellow_upper = np.array([35, 255, 255])
    
    orange_lower = np.array([8, 100, 100])
    orange_upper = np.array([20, 255, 255])
    
    brown_lower = np.array([8, 40, 20])
    brown_upper = np.array([18, 200, 120])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & mango_mask
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) & mango_mask
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper) & mango_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & mango_mask
    
    total_valid = np.count_nonzero(mango_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'yellow_ratio': np.count_nonzero(yellow_mask) / max(total_valid, 1),
        'orange_ratio': np.count_nonzero(orange_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_mango_ripeness(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    orange_ratio = features['orange_ratio']
    brown_ratio = features['brown_ratio']
    
    ripe_ratio = yellow_ratio + orange_ratio
    
    if green_ratio > 0.5:
        confidence = min(90, 65 + green_ratio * 45)
        return "Unripe", int(confidence)
    
    elif ripe_ratio > 0.4 and brown_ratio < 0.1:
        confidence = min(95, 70 + ripe_ratio * 40)
        return "Ripe", int(confidence)
    
    elif brown_ratio > 0.15:
        confidence = min(88, 60 + brown_ratio * 70)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 72

def get_mango_color(features):
    green_ratio = features['green_ratio']
    ripe_ratio = features['yellow_ratio'] + features['orange_ratio']
    
    if green_ratio > 0.4:
        return "Green"
    elif features['orange_ratio'] > features['yellow_ratio']:
        return "Orange-Yellow"
    else:
        return "Yellow"

def detect_mango_quality(features):
    brown_ratio = features['brown_ratio']
    ripe_ratio = features['yellow_ratio'] + features['orange_ratio']
    
    if ripe_ratio > 0.5 and brown_ratio < 0.05:
        return "Grade A (Premium)"
    elif brown_ratio < 0.12:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== ORANGE DETECTION ==========
def extract_orange_features(hsv, bgr_image):
    """Extract color features from orange"""
    h, s, v = cv2.split(hsv)
    
    orange_mask = (s > 25) & (v > 40) & (v < 250)
    
    green_lower = np.array([35, 30, 30])
    green_upper = np.array([85, 255, 255])
    
    orange_lower = np.array([8, 100, 100])
    orange_upper = np.array([20, 255, 255])
    
    brown_lower = np.array([8, 40, 20])
    brown_upper = np.array([18, 180, 100])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & orange_mask
    orange_mask_color = cv2.inRange(hsv, orange_lower, orange_upper) & orange_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & orange_mask
    
    total_valid = np.count_nonzero(orange_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'orange_ratio': np.count_nonzero(orange_mask_color) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_orange_ripeness(features):
    green_ratio = features['green_ratio']
    orange_ratio = features['orange_ratio']
    brown_ratio = features['brown_ratio']
    
    if green_ratio > 0.4:
        confidence = min(88, 65 + green_ratio * 40)
        return "Unripe", int(confidence)
    
    elif orange_ratio > 0.4 and brown_ratio < 0.1:
        confidence = min(93, 70 + orange_ratio * 45)
        return "Ripe", int(confidence)
    
    elif brown_ratio > 0.12:
        confidence = min(85, 60 + brown_ratio * 65)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 68

def get_orange_color(features):
    green_ratio = features['green_ratio']
    orange_ratio = features['orange_ratio']
    
    if orange_ratio > 0.4:
        return "Orange"
    elif green_ratio > 0.3:
        return "Green"
    else:
        return "Orange-Green"

def detect_orange_quality(features):
    brown_ratio = features['brown_ratio']
    orange_ratio = features['orange_ratio']
    
    if orange_ratio > 0.5 and brown_ratio < 0.05:
        return "Grade A (Premium)"
    elif brown_ratio < 0.10:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== TOMATO DETECTION ==========
def extract_tomato_features(hsv, bgr_image):
    """Extract color features from tomato"""
    h, s, v = cv2.split(hsv)
    
    tomato_mask = (s > 20) & (v > 30) & (v < 250)
    
    green_lower = np.array([35, 30, 30])
    green_upper = np.array([85, 255, 255])
    
    red_lower1 = np.array([0, 70, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 70])
    red_upper2 = np.array([180, 255, 255])
    
    orange_lower = np.array([10, 70, 70])
    orange_upper = np.array([20, 255, 255])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & tomato_mask
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1) & tomato_mask
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2) & tomato_mask
    red_mask = red_mask1 | red_mask2
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper) & tomato_mask
    
    total_valid = np.count_nonzero(tomato_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'red_ratio': np.count_nonzero(red_mask) / max(total_valid, 1),
        'orange_ratio': np.count_nonzero(orange_mask) / max(total_valid, 1),
    }
    
    return features

def detect_tomato_ripeness(features):
    green_ratio = features['green_ratio']
    red_ratio = features['red_ratio']
    orange_ratio = features['orange_ratio']
    
    if green_ratio > 0.5:
        confidence = min(92, 68 + green_ratio * 42)
        return "Unripe", int(confidence)
    
    elif red_ratio > 0.5:
        confidence = min(95, 72 + red_ratio * 45)
        return "Ripe", int(confidence)
    
    elif orange_ratio > 0.3 or (red_ratio > 0.2 and green_ratio > 0.2):
        return "Turning Ripe", 74
    
    elif red_ratio > 0.6:
        return "Overripe", 80
    
    else:
        return "Turning Ripe", 65

def get_tomato_color(features):
    green_ratio = features['green_ratio']
    red_ratio = features['red_ratio']
    
    if red_ratio > 0.5:
        return "Red"
    elif green_ratio > 0.4:
        return "Green"
    else:
        return "Red-Orange"

def detect_tomato_quality(features):
    red_ratio = features['red_ratio']
    green_ratio = features['green_ratio']
    
    if red_ratio > 0.6 and green_ratio < 0.15:
        return "Grade A (Premium)"
    elif red_ratio > 0.3:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== AI RECOMMENDATIONS ==========
def generate_recommendation(fruit_type, ripeness, quality, confidence):
    """Generate smart eating/usage recommendations"""
    
    fruit_emoji = {
        "banana": "🍌",
        "apple": "🍎",
        "mango": "🥭",
        "orange": "🍊",
        "tomato": "🍅",
        "strawberry": "🍓",
        "grapes": "🍇",
        "avocado": "🥑",
        "pineapple": "🍍",
        "papaya": "🫐"
    }
    
    emoji = fruit_emoji.get(fruit_type.lower(), "🍎")
    
    # Low confidence warning
    if confidence < 60:
        return f"⚠️ Low confidence detection. Please try another photo with better lighting and centered {fruit_type}."
    
    # Recommendations based on ripeness
    if "Unripe" in ripeness:
        if fruit_type.lower() == "avocado":
            return f"{emoji} Not ready yet! Leave at room temp for 2-5 days. Speed up: Put in paper bag with apple!"
        elif fruit_type.lower() == "pineapple":
            return f"{emoji} Not ready! Wait 2-3 days at room temp. Bottom should smell sweet when ripe."
        elif fruit_type.lower() == "papaya":
            return f"{emoji} Not ready yet! Keep at room temp for 3-5 days until yellow-orange."
        elif fruit_type.lower() in ["banana", "mango"]:
            return f"{emoji} Not ready yet! Wait 2-3 days for best taste. Keep at room temperature."
        elif fruit_type.lower() == "tomato":
            return f"{emoji} Not ready yet! Leave on counter for 2-4 days until red. Don't refrigerate!"
        elif fruit_type.lower() == "strawberry":
            return f"{emoji} Too early! Strawberries don't ripen after picking. Choose redder ones next time."
        else:
            return f"{emoji} Not ready to eat. Wait a few days for better flavor and sweetness."
    
    elif "Turning" in ripeness:
        if fruit_type.lower() == "avocado":
            return f"{emoji} Almost there! 1-2 days until perfect. Check daily by gently squeezing."
        elif fruit_type.lower() == "banana":
            return f"{emoji} Almost there! Perfect in 1-2 days. Great for cereal or smoothies now."
        else:
            return f"{emoji} Getting close! Will be perfect in 1-2 days."
    
    elif "Ripe" in ripeness and "Over" not in ripeness:
        if fruit_type.lower() == "avocado":
            return f"✨ PERFECT! {emoji} Ready for guacamole, toast, or salads. Use within 2 days!"
        elif fruit_type.lower() == "strawberry":
            return f"✨ PERFECT! {emoji} Sweet and juicy! Eat fresh or use in desserts. Best within 2 days!"
        elif fruit_type.lower() == "grapes":
            return f"✨ PERFECT! {emoji} Sweet and crisp! Refrigerate to keep fresh longer."
        elif fruit_type.lower() == "pineapple":
            return f"✨ PERFECT! {emoji} Sweet and juicy! Great fresh or in smoothies!"
        elif "Grade A" in quality:
            return f"✨ PERFECT! {emoji} Best time to eat - sweet, flavorful, and premium quality!"
        elif "Grade B" in quality:
            return f"👍 Great! {emoji} Perfect for eating now. Enjoy fresh or in salads!"
        else:
            return f"{emoji} Ready to eat! Best consumed soon or use in cooking."
    
    elif "Overripe" in ripeness:
        if fruit_type.lower() == "banana":
            return f"🥞 Overripe for fresh eating. Perfect for banana bread, smoothies, or baking!"
        elif fruit_type.lower() == "avocado":
            return f"⚠️ Past prime! Check for brown spots inside. Use immediately in guacamole or dips."
        elif fruit_type.lower() == "strawberry":
            return f"🍰 Too soft for fresh eating. Make jam, sauce, or use in baking!"
        elif fruit_type.lower() == "tomato":
            return f"🍝 Best for cooking! Use in pasta sauce, soups, or cooked dishes."
        elif fruit_type.lower() == "apple":
            return f"🥧 Best for cooking! Make applesauce, pie, or baked apples."
        elif fruit_type.lower() in ["pineapple", "papaya", "mango"]:
            return f"🍹 Too soft for slicing. Perfect for smoothies, juice, or cooked dishes!"
        else:
            return f"🍳 Past prime for fresh eating. Great for cooking, juice, or smoothies!"
    
    else:
        return f"{emoji} Quality detected. Store properly to maintain freshness!"
    
    # ========== STRAWBERRY DETECTION ==========
def extract_strawberry_features(hsv, bgr_image):
    """Extract color features from strawberry"""
    h, s, v = cv2.split(hsv)
    
    strawberry_mask = (s > 30) & (v > 40) & (v < 250)
    
    # White/pale (unripe)
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 50, 255])
    
    # Pink (turning ripe)
    pink_lower = np.array([160, 50, 100])
    pink_upper = np.array([180, 255, 255])
    
    # Red (ripe)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    # Dark red/brown (overripe)
    dark_lower = np.array([0, 50, 30])
    dark_upper = np.array([10, 255, 100])
    
    white_mask = cv2.inRange(hsv, white_lower, white_upper) & strawberry_mask
    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper) & strawberry_mask
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1) & strawberry_mask
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2) & strawberry_mask
    red_mask = red_mask1 | red_mask2
    dark_mask = cv2.inRange(hsv, dark_lower, dark_upper) & strawberry_mask
    
    total_valid = np.count_nonzero(strawberry_mask)
    
    features = {
        'white_ratio': np.count_nonzero(white_mask) / max(total_valid, 1),
        'pink_ratio': np.count_nonzero(pink_mask) / max(total_valid, 1),
        'red_ratio': np.count_nonzero(red_mask) / max(total_valid, 1),
        'dark_ratio': np.count_nonzero(dark_mask) / max(total_valid, 1),
    }
    
    return features

def detect_strawberry_ripeness(features):
    white_ratio = features['white_ratio']
    red_ratio = features['red_ratio']
    dark_ratio = features['dark_ratio']
    
    if white_ratio > 0.3:
        confidence = min(88, 65 + white_ratio * 45)
        return "Unripe", int(confidence)
    
    elif red_ratio > 0.5 and dark_ratio < 0.1:
        confidence = min(93, 70 + red_ratio * 45)
        return "Ripe", int(confidence)
    
    elif dark_ratio > 0.15:
        confidence = min(85, 60 + dark_ratio * 60)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 72

def get_strawberry_color(features):
    red_ratio = features['red_ratio']
    white_ratio = features['white_ratio']
    
    if red_ratio > 0.5:
        return "Red"
    elif white_ratio > 0.3:
        return "White-Pink"
    else:
        return "Pink-Red"

def detect_strawberry_quality(features):
    dark_ratio = features['dark_ratio']
    red_ratio = features['red_ratio']
    
    if red_ratio > 0.6 and dark_ratio < 0.05:
        return "Grade A (Premium)"
    elif dark_ratio < 0.12:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== GRAPES DETECTION ==========
def extract_grapes_features(hsv, bgr_image):
    """Extract color features from grapes"""
    h, s, v = cv2.split(hsv)
    
    grapes_mask = (s > 20) & (v > 30) & (v < 250)
    
    # Green grapes
    green_lower = np.array([35, 30, 50])
    green_upper = np.array([85, 255, 255])
    
    # Purple/red grapes
    purple_lower1 = np.array([140, 50, 50])
    purple_upper1 = np.array([180, 255, 255])
    purple_lower2 = np.array([0, 50, 50])
    purple_upper2 = np.array([10, 255, 255])
    
    # Brown (overripe)
    brown_lower = np.array([8, 40, 20])
    brown_upper = np.array([20, 200, 120])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & grapes_mask
    purple_mask1 = cv2.inRange(hsv, purple_lower1, purple_upper1) & grapes_mask
    purple_mask2 = cv2.inRange(hsv, purple_lower2, purple_upper2) & grapes_mask
    purple_mask = purple_mask1 | purple_mask2
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & grapes_mask
    
    total_valid = np.count_nonzero(grapes_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'purple_ratio': np.count_nonzero(purple_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_grapes_ripeness(features):
    green_ratio = features['green_ratio']
    purple_ratio = features['purple_ratio']
    brown_ratio = features['brown_ratio']
    
    if brown_ratio > 0.2:
        confidence = min(85, 60 + brown_ratio * 65)
        return "Overripe", int(confidence)
    
    elif (green_ratio > 0.4 or purple_ratio > 0.4) and brown_ratio < 0.1:
        confidence = min(92, 70 + max(green_ratio, purple_ratio) * 45)
        return "Ripe", int(confidence)
    
    elif green_ratio < 0.2 and purple_ratio < 0.2:
        return "Unripe", 68
    
    else:
        return "Turning Ripe", 73

def get_grapes_color(features):
    green_ratio = features['green_ratio']
    purple_ratio = features['purple_ratio']
    
    if purple_ratio > green_ratio and purple_ratio > 0.3:
        return "Purple/Red"
    elif green_ratio > 0.3:
        return "Green"
    else:
        return "Mixed"

def detect_grapes_quality(features):
    brown_ratio = features['brown_ratio']
    
    if brown_ratio < 0.05:
        return "Grade A (Premium)"
    elif brown_ratio < 0.15:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== AVOCADO DETECTION ==========
def extract_avocado_features(hsv, bgr_image):
    """Extract color features from avocado"""
    h, s, v = cv2.split(hsv)
    
    avocado_mask = (s > 15) & (v > 20) & (v < 250)
    
    # Bright green (unripe)
    bright_green_lower = np.array([40, 50, 80])
    bright_green_upper = np.array([80, 255, 255])
    
    # Dark green (ripe)
    dark_green_lower = np.array([35, 40, 30])
    dark_green_upper = np.array([85, 255, 120])
    
    # Very dark/black (perfect ripe)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 50])
    
    bright_green_mask = cv2.inRange(hsv, bright_green_lower, bright_green_upper) & avocado_mask
    dark_green_mask = cv2.inRange(hsv, dark_green_lower, dark_green_upper) & avocado_mask
    black_mask = cv2.inRange(hsv, black_lower, black_upper) & avocado_mask
    
    total_valid = np.count_nonzero(avocado_mask)
    
    features = {
        'bright_green_ratio': np.count_nonzero(bright_green_mask) / max(total_valid, 1),
        'dark_green_ratio': np.count_nonzero(dark_green_mask) / max(total_valid, 1),
        'black_ratio': np.count_nonzero(black_mask) / max(total_valid, 1),
        'mean_value': np.mean(v[avocado_mask]) if total_valid > 0 else 0,
    }
    
    return features

def detect_avocado_ripeness(features):
    bright_green_ratio = features['bright_green_ratio']
    dark_green_ratio = features['dark_green_ratio']
    black_ratio = features['black_ratio']
    mean_value = features['mean_value']
    
    if bright_green_ratio > 0.4 or mean_value > 120:
        confidence = min(88, 65 + bright_green_ratio * 40)
        return "Unripe", int(confidence)
    
    elif (dark_green_ratio > 0.3 or black_ratio > 0.2) and mean_value < 80:
        confidence = min(92, 70 + (dark_green_ratio + black_ratio) * 35)
        return "Ripe", int(confidence)
    
    elif black_ratio > 0.4:
        return "Overripe", 75
    
    else:
        return "Turning Ripe", 70

def get_avocado_color(features):
    bright_green_ratio = features['bright_green_ratio']
    black_ratio = features['black_ratio']
    
    if bright_green_ratio > 0.4:
        return "Bright Green"
    elif black_ratio > 0.3:
        return "Dark/Black"
    else:
        return "Dark Green"

def detect_avocado_quality(features):
    black_ratio = features['black_ratio']
    dark_green_ratio = features['dark_green_ratio']
    
    if (dark_green_ratio > 0.3 or black_ratio > 0.2) and black_ratio < 0.5:
        return "Grade A (Premium)"
    elif black_ratio < 0.6:
        return "Grade B (Good)"
    else:
        return "Grade C (Overripe)"


# ========== PINEAPPLE DETECTION ==========
def extract_pineapple_features(hsv, bgr_image):
    """Extract color features from pineapple"""
    h, s, v = cv2.split(hsv)
    
    pineapple_mask = (s > 25) & (v > 40) & (v < 250)
    
    # Green (unripe)
    green_lower = np.array([35, 40, 40])
    green_upper = np.array([85, 255, 255])
    
    # Yellow-orange (ripe)
    yellow_lower = np.array([18, 80, 100])
    yellow_upper = np.array([35, 255, 255])
    
    # Brown (overripe)
    brown_lower = np.array([8, 50, 30])
    brown_upper = np.array([20, 255, 150])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & pineapple_mask
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) & pineapple_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & pineapple_mask
    
    total_valid = np.count_nonzero(pineapple_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'yellow_ratio': np.count_nonzero(yellow_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_pineapple_ripeness(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    brown_ratio = features['brown_ratio']
    
    if green_ratio > 0.5:
        confidence = min(87, 65 + green_ratio * 40)
        return "Unripe", int(confidence)
    
    elif yellow_ratio > 0.4 and brown_ratio < 0.15:
        confidence = min(91, 68 + yellow_ratio * 42)
        return "Ripe", int(confidence)
    
    elif brown_ratio > 0.2:
        confidence = min(83, 58 + brown_ratio * 55)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 71

def get_pineapple_color(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    
    if yellow_ratio > 0.4:
        return "Golden Yellow"
    elif green_ratio > 0.4:
        return "Green"
    else:
        return "Yellow-Green"

def detect_pineapple_quality(features):
    brown_ratio = features['brown_ratio']
    yellow_ratio = features['yellow_ratio']
    
    if yellow_ratio > 0.5 and brown_ratio < 0.08:
        return "Grade A (Premium)"
    elif brown_ratio < 0.18:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"


# ========== PAPAYA DETECTION ==========
def extract_papaya_features(hsv, bgr_image):
    """Extract color features from papaya"""
    h, s, v = cv2.split(hsv)
    
    papaya_mask = (s > 20) & (v > 35) & (v < 250)
    
    # Green (unripe)
    green_lower = np.array([35, 35, 35])
    green_upper = np.array([85, 255, 255])
    
    # Yellow (ripe)
    yellow_lower = np.array([20, 60, 100])
    yellow_upper = np.array([35, 255, 255])
    
    # Orange (very ripe)
    orange_lower = np.array([8, 100, 100])
    orange_upper = np.array([20, 255, 255])
    
    # Brown (overripe)
    brown_lower = np.array([8, 40, 20])
    brown_upper = np.array([18, 200, 120])
    
    green_mask = cv2.inRange(hsv, green_lower, green_upper) & papaya_mask
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper) & papaya_mask
    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper) & papaya_mask
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper) & papaya_mask
    
    total_valid = np.count_nonzero(papaya_mask)
    
    features = {
        'green_ratio': np.count_nonzero(green_mask) / max(total_valid, 1),
        'yellow_ratio': np.count_nonzero(yellow_mask) / max(total_valid, 1),
        'orange_ratio': np.count_nonzero(orange_mask) / max(total_valid, 1),
        'brown_ratio': np.count_nonzero(brown_mask) / max(total_valid, 1),
    }
    
    return features

def detect_papaya_ripeness(features):
    green_ratio = features['green_ratio']
    yellow_ratio = features['yellow_ratio']
    orange_ratio = features['orange_ratio']
    brown_ratio = features['brown_ratio']
    
    ripe_ratio = yellow_ratio + orange_ratio
    
    if green_ratio > 0.5:
        confidence = min(89, 66 + green_ratio * 43)
        return "Unripe", int(confidence)
    
    elif ripe_ratio > 0.4 and brown_ratio < 0.12:
        confidence = min(93, 69 + ripe_ratio * 44)
        return "Ripe", int(confidence)
    
    elif brown_ratio > 0.18:
        confidence = min(86, 60 + brown_ratio * 62)
        return "Overripe", int(confidence)
    
    else:
        return "Turning Ripe", 72

def get_papaya_color(features):
    green_ratio = features['green_ratio']
    ripe_ratio = features['yellow_ratio'] + features['orange_ratio']
    
    if green_ratio > 0.4:
        return "Green"
    elif features['orange_ratio'] > features['yellow_ratio']:
        return "Orange-Yellow"
    else:
        return "Yellow"

def detect_papaya_quality(features):
    brown_ratio = features['brown_ratio']
    ripe_ratio = features['yellow_ratio'] + features['orange_ratio']
    
    if ripe_ratio > 0.5 and brown_ratio < 0.06:
        return "Grade A (Premium)"
    elif brown_ratio < 0.15:
        return "Grade B (Good)"
    else:
        return "Grade C (Fair)"