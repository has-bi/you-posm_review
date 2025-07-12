# app.py
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageStat, ImageFilter
import base64
import io
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(
    page_title="COC Shelf Reviewer POC",
    page_icon="🏪",
    layout="wide"
)

def check_image_quality(image: Image) -> dict:
    """Check image quality metrics using PIL only"""
    
    # 1. Check blur using PIL edge detection
    gray_image = image.convert('L')
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    blur_score = np.var(edge_array)
    is_blurry = blur_score < 500  # Adjusted threshold for PIL method
    
    # 2. Check brightness
    stat = ImageStat.Stat(image)
    brightness = stat.mean[0] if len(stat.mean) > 0 else sum(stat.mean) / len(stat.mean)
    is_too_dark = brightness < 60
    is_too_bright = brightness > 200
    
    # 3. Check resolution
    width, height = image.size
    is_low_res = width < 800 or height < 600
    
    # 4. Check aspect ratio (should be portrait for shelf)
    aspect_ratio = height / width
    is_good_orientation = 1.2 < aspect_ratio < 2.0
    
    # 5. Additional quality checks
    # Check if image is too uniform (might indicate poor capture)
    std_dev = stat.stddev[0] if len(stat.stddev) > 0 else sum(stat.stddev) / len(stat.stddev)
    is_too_uniform = std_dev < 20
    
    return {
        'is_blurry': is_blurry,
        'blur_score': blur_score,
        'is_too_dark': is_too_dark,
        'is_too_bright': is_too_bright,
        'brightness': brightness,
        'is_low_res': is_low_res,
        'resolution': f"{width}x{height}",
        'is_good_orientation': is_good_orientation,
        'aspect_ratio': aspect_ratio,
        'is_too_uniform': is_too_uniform,
        'std_dev': std_dev,
        'overall_quality': not (is_blurry or is_too_dark or is_too_bright or is_low_res or is_too_uniform)
    }

def validate_image_consistency(img1: Image, img2: Image, img3: Image) -> dict:
    """Check if images are taken from similar positions"""
    
    # Check if resolutions are similar
    sizes = [img.size for img in [img1, img2, img3]]
    max_size_diff = max(
        abs(s1[0] - s2[0]) / max(s1[0], 1) 
        for s1, s2 in zip(sizes, sizes[1:])
    )
    
    # Check aspect ratios
    ratios = [s[0]/s[1] for s in sizes]
    max_ratio_diff = max(ratios) - min(ratios)
    
    # Check if all images have similar brightness (consistency indicator)
    brightnesses = []
    for img in [img1, img2, img3]:
        stat = ImageStat.Stat(img)
        brightness = stat.mean[0] if len(stat.mean) > 0 else sum(stat.mean) / len(stat.mean)
        brightnesses.append(brightness)
    
    brightness_consistency = max(brightnesses) - min(brightnesses) < 50
    
    return {
        'consistent_size': max_size_diff < 0.2,  # 20% tolerance
        'consistent_ratio': max_ratio_diff < 0.3,
        'consistent_lighting': brightness_consistency,
        'size_difference': f"{max_size_diff:.1%}",
        'sizes': sizes,
        'brightnesses': brightnesses
    }

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    # Resize if too large
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def build_quality_aware_prompt(shelf_position, quality_issues):
    """Build prompt considering image quality and Youvit focus"""
    
    shelf_names = {1: "Top", 2: "Second", 3: "Third", 4: "Bottom"}
    
    prompt = f"""
    Analyze this shelf #{shelf_position} ({shelf_names[shelf_position]} shelf) focusing ONLY on Youvit brand products.
    
    BRAND IDENTIFICATION:
    - Youvit products have "YOUVIT" text/logo on packaging
    - Common variants: Youvit Adult Multivitamin, Youvit Kids Multivitamin, Youvit Vision Kids, Youvit Collagen
    - Usually have white or colorful packaging with distinctive branding
    """
    
    # Add quality warnings
    if quality_issues.get('has_blur'):
        prompt += "\n\nNOTE: Some images are blurry. Only analyze clearly visible products."
    
    if quality_issues.get('has_lighting_issues'):
        prompt += "\n\nNOTE: Lighting issues detected. Be cautious about color-based identification."
    
    if quality_issues.get('inconsistent_angles'):
        prompt += "\n\nNOTE: Images taken from different angles. Focus on obvious improvements."
    
    prompt += f"""
    
    ANALYSIS METHOD:
    1. First identify all Youvit products in each image (or note if none visible)
    2. List all Youvit product names found in BEFORE and AFTER images
    3. Compare BEFORE vs AFTER to measure merchandiser's improvement work
    4. Check AFTER against IDEAL for planogram compliance
    
    EVALUATION CRITERIA:
    - Did merchandiser improve Youvit product visibility?
    - Are Youvit products properly faced and organized?
    - Is Youvit section clean and well-blocked?
    - Are competitor products removed from Youvit space?
    - Is the Youvit product stock level sufficient?
    
    IMPORTANT:
    - Only score based on Youvit products
    - If no Youvit products clearly visible, state this
    - Focus on actual work done by merchandiser
    - Note confidence level (HIGH/MEDIUM/LOW) for findings
    
    Return ONLY valid JSON:
    {{
        "overallConfidence": "HIGH/MEDIUM/LOW",
        "youvitProductsFound": {{
            "before": ["list of Youvit products identified"],
            "after": ["list of Youvit products identified"],
            "ideal": ["list of Youvit products in ideal"]
        }},
        "overallScore": 0-100,
        "improvementScore": 0-100,
        "complianceScore": 0-100,
        "details": {{
            "productFacing": {{"score": 0-100, "issues": [], "confidence": "HIGH/MEDIUM/LOW"}},
            "arrangement": {{"score": 0-100, "issues": [], "confidence": "HIGH/MEDIUM/LOW"}},
            "stockLevel": {{"score": 0-100, "issues": [], "confidence": "HIGH/MEDIUM/LOW"}},
            "brandBlocking": {{"score": 0-100, "issues": [], "confidence": "HIGH/MEDIUM/LOW"}},
            "cleanliness": {{"score": 0-100, "issues": [], "confidence": "HIGH/MEDIUM/LOW"}}
        }},
        "improvements": ["specific improvements made by merchandiser"],
        "stillNeedsFixing": ["remaining issues"],
        "criticalFindings": ["urgent issues if any"]
    }}
    """
    
    return prompt

def analyze_shelf(ideal_img, before_img, after_img, shelf_position, quality_issues):
    """Call GPT-4 Vision API for analysis"""
    try:
        # Convert images to base64
        with st.spinner("Preparing images..."):
            ideal_b64 = image_to_base64(ideal_img)
            before_b64 = image_to_base64(before_img)
            after_b64 = image_to_base64(after_img)
        
        # Build quality-aware prompt
        prompt = build_quality_aware_prompt(shelf_position, quality_issues)
        
        # Call GPT-4 Vision
        with st.spinner("Analyzing with GPT-4 Vision..."):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": ideal_b64, "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": before_b64, "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": after_b64, "detail": "high"}}
                    ]
                }],
                max_tokens=1500,
                temperature=0.1
            )
        
        # Parse response
        result_text = response.choices[0].message.content
        
        # Clean up response if it has markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(result_text)
        
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse GPT response as JSON: {e}")
        st.text("Raw response:")
        st.code(result_text)
        return None
    except Exception as e:
        st.error(f"Error analyzing images: {str(e)}")
        return None

def validate_uploaded_images(ideal_img, before_img, after_img):
    """Comprehensive image validation"""
    
    quality_results = {}
    all_good = True
    quality_issues = {}
    
    # Check each image
    for name, img in [("Ideal", ideal_img), ("Before", before_img), ("After", after_img)]:
        quality = check_image_quality(img)
        quality_results[name] = quality
        
        if not quality['overall_quality']:
            all_good = False
    
    # Check consistency
    consistency = validate_image_consistency(ideal_img, before_img, after_img)
    
    # Determine quality issues for prompt
    quality_issues['has_blur'] = any(q['is_blurry'] for q in quality_results.values())
    quality_issues['has_lighting_issues'] = any(
        q['is_too_dark'] or q['is_too_bright'] 
        for q in quality_results.values()
    )
    quality_issues['inconsistent_angles'] = not consistency['consistent_size']
    
    # Display validation results
    with st.expander("🔍 Image Quality Check", expanded=not all_good):
        # Overall status
        if all_good and consistency['consistent_size']:
            st.success("✅ All images pass quality checks!")
        else:
            st.warning("⚠️ Some quality issues detected - results may be less accurate")
        
        # Individual image quality
        cols = st.columns(3)
        for idx, (name, quality) in enumerate(quality_results.items()):
            with cols[idx]:
                st.markdown(f"**{name} Image**")
                
                # Quality indicators
                indicators = []
                if quality['is_blurry']:
                    indicators.append("🔍 Blurry")
                if quality['is_too_dark']:
                    indicators.append("🌑 Too Dark")
                if quality['is_too_bright']:
                    indicators.append("☀️ Too Bright")
                if quality['is_low_res']:
                    indicators.append("📐 Low Resolution")
                if not quality['is_good_orientation']:
                    indicators.append("🔄 Wrong Orientation")
                if quality['is_too_uniform']:
                    indicators.append("📊 Low Contrast")
                
                if indicators:
                    for indicator in indicators:
                        st.caption(indicator)
                else:
                    st.caption("✅ Good Quality")
                
                # Metrics
                st.caption(f"Resolution: {quality['resolution']}")
                st.caption(f"Blur Score: {quality['blur_score']:.0f}")
                st.caption(f"Brightness: {quality['brightness']:.0f}")
                st.caption(f"Contrast: {quality['std_dev']:.0f}")
        
        # Consistency check
        inconsistencies = []
        if not consistency['consistent_size']:
            inconsistencies.append(f"📐 Size difference: {consistency['size_difference']}")
        if not consistency['consistent_lighting']:
            inconsistencies.append("💡 Lighting inconsistency")
        
        if inconsistencies:
            for issue in inconsistencies:
                st.error(issue)
    
    return quality_issues

# def show_capture_guidelines():
#     """Display image capture best practices in sidebar"""
#     st.sidebar.markdown("""
#     ## 📸 Photo Capture Guide
    
#     ### ✅ DO:
#     - Stand **1-1.5 meters** from shelf
#     - **Center** on Youvit section
#     - Keep phone **parallel** to shelf
#     - Use **landscape** orientation
#     - Ensure **good lighting**
#     - Take all photos from **same position**
    
#     ### ❌ DON'T:
#     - Take from angle/side
#     - Use different distances
#     - Cut off products
#     - Use flash (causes glare)
#     - Include people/carts
#     - Use portrait mode
    
#     ### 💡 Quick Check:
#     Can you clearly read "YOUVIT" on products?
#     """)

# Main UI
st.title("🏪 COC Shelf Reviewer - Youvit Focus")
st.markdown("Analyze Youvit product placement using GPT-4 Vision")

# Show capture guidelines in sidebar
# show_capture_guidelines()

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Configuration")
    
    # Store code input
    store_code = st.text_input("Store Code", value="JKT001", help="Enter store identifier")
    
    # Shelf position selector
    shelf_position = st.selectbox(
        "Shelf Position to Analyze",
        options=[1, 2, 3, 4],
        format_func=lambda x: f"Shelf {x} - {['Top', 'Second', 'Third', 'Bottom'][x-1]}"
    )
    
    st.subheader("Upload Images")
    st.caption("Ensure Youvit products are clearly visible")
    
    # Image uploaders
    ideal_file = st.file_uploader("📋 Ideal Reference", type=['jpg', 'jpeg', 'png'])
    before_file = st.file_uploader("❌ Before Image", type=['jpg', 'jpeg', 'png'])
    after_file = st.file_uploader("✅ After Image", type=['jpg', 'jpeg', 'png'])
    
    # Ready check
    ready_to_analyze = ideal_file and before_file and after_file

with col2:
    if ready_to_analyze:
        # Load images
        ideal_img = Image.open(ideal_file)
        before_img = Image.open(before_file)
        after_img = Image.open(after_file)
        
        # Display images
        st.subheader("Image Preview")
        img_col1, img_col2, img_col3 = st.columns(3)
        
        with img_col1:
            st.image(ideal_img, caption="📋 Ideal Reference", use_container_width=True)
        
        with img_col2:
            st.image(before_img, caption="❌ Before", use_container_width=True)
            
        with img_col3:
            st.image(after_img, caption="✅ After", use_container_width=True)
        
        # Validate images
        quality_issues = validate_uploaded_images(ideal_img, before_img, after_img)
        
        # Analyze button
        if st.button("🔍 Analyze Youvit Products", type="primary", use_container_width=True):
            # Run analysis
            analysis = analyze_shelf(ideal_img, before_img, after_img, shelf_position, quality_issues)
            
            if analysis:
                # Check if Youvit products were found
                youvit_found = analysis.get('youvitProductsFound', {})
                
                if not youvit_found.get('after', []):
                    st.error("❌ No Youvit products detected in images. Please ensure Youvit products are clearly visible.")
                else:
                    st.success("✅ Analysis Complete!")
                    
                    # Display confidence level
                    confidence = analysis.get('overallConfidence', 'MEDIUM')
                    conf_color = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}
                    st.info(f"Analysis Confidence: {conf_color.get(confidence, '🟡')} {confidence}")
                    
                    # Display Youvit products found
                    with st.expander("🔍 Youvit Products Detected"):
                        col_b, col_a, col_i = st.columns(3)
                        with col_b:
                            st.markdown("**Before:**")
                            for product in youvit_found.get('before', []):
                                st.caption(f"• {product}")
                        with col_a:
                            st.markdown("**After:**")
                            for product in youvit_found.get('after', []):
                                st.caption(f"• {product}")
                        with col_i:
                            st.markdown("**Ideal:**")
                            for product in youvit_found.get('ideal', []):
                                st.caption(f"• {product}")
                    
                    # Display scores
                    st.subheader("Analysis Results")
                    
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        overall = analysis.get('overallScore', 0)
                        st.metric("Overall Score", f"{overall}%", 
                                 help="Combined score of all factors")
                    
                    with score_cols[1]:
                        improvement = analysis.get('improvementScore', 0)
                        st.metric("Improvement Score", f"{improvement}%",
                                 help="How much better is After vs Before")
                    
                    with score_cols[2]:
                        compliance = analysis.get('complianceScore', 0)
                        st.metric("Compliance Score", f"{compliance}%",
                                 help="How well After matches Ideal")
                    
                    # Progress bar for overall score
                    st.progress(overall / 100)
                    
                    # Detailed breakdown
                    st.subheader("Detailed Analysis")
                    
                    categories = [
                        ('productFacing', 'Product Facing', '📦'),
                        ('arrangement', 'Arrangement', '📊'),
                        ('stockLevel', 'Stock Level', '📈'),
                        ('brandBlocking', 'Brand Blocking', '🔲'),
                        ('cleanliness', 'Cleanliness', '✨')
                    ]
                    
                    for cat_key, cat_name, icon in categories:
                        if cat_key in analysis.get('details', {}):
                            cat_data = analysis['details'][cat_key]
                            cat_score = cat_data.get('score', 0)
                            cat_conf = cat_data.get('confidence', 'MEDIUM')
                            issues = cat_data.get('issues', [])
                            
                            with st.expander(f"{icon} {cat_name} - {cat_score}% ({cat_conf})"):
                                st.progress(cat_score / 100)
                                if issues:
                                    for issue in issues:
                                        st.write(f"• {issue}")
                                else:
                                    st.write("✅ No issues found")
                    
                    # Improvements and remaining issues
                    col_imp, col_fix = st.columns(2)
                    
                    with col_imp:
                        st.subheader("✅ Improvements Made")
                        improvements = analysis.get('improvements', [])
                        if improvements:
                            for improvement in improvements:
                                st.success(f"• {improvement}")
                        else:
                            st.info("No significant improvements detected")
                    
                    with col_fix:
                        st.subheader("⚠️ Still Needs Fixing")
                        issues = analysis.get('stillNeedsFixing', [])
                        if issues:
                            for issue in issues:
                                st.warning(f"• {issue}")
                        else:
                            st.info("All issues resolved!")
                    
                    # Critical findings
                    critical = analysis.get('criticalFindings', [])
                    if critical:
                        st.error("🚨 **Critical Issues:**")
                        for finding in critical:
                            st.error(f"• {finding}")
                    
                    # Export results
                    st.subheader("Export Results")
                    
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "store_code": store_code,
                        "shelf_position": shelf_position,
                        "quality_issues": quality_issues,
                        "analysis": analysis
                    }
                    
                    st.download_button(
                        label="📥 Download Full Report (JSON)",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"youvit_shelf_analysis_{store_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Save to session state for history
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'store_code': store_code,
                        'shelf_position': shelf_position,
                        'score': overall,
                        'confidence': confidence
                    })
                    
    else:
        st.info("👆 Please upload all three images to begin analysis")
        
        # Show example
        with st.expander("📖 See Example"):
            st.markdown("""
            **What makes a good photo:**
            - Youvit products clearly visible with readable logos
            - Entire shelf section in frame
            - Good lighting without shadows
            - Taken straight-on (not from angle)
            - All 3 photos from same position
            """)

# Footer
st.markdown("---")
st.caption("COC Shelf Reviewer POC v0.1.0 - Focused on Youvit Products")

# History in sidebar
if st.sidebar.checkbox("Show History"):
    st.sidebar.subheader("Recent Analyses")
    if 'history' in st.session_state and st.session_state.history:
        for record in reversed(st.session_state.history[-5:]):
            st.sidebar.caption(
                f"{record['store_code']} - Shelf {record['shelf_position']} - "
                f"{record['score']}% ({record['confidence']}) - "
                f"{record['timestamp'].strftime('%H:%M')}"
            )