# app.py - Batched Analysis Unit Test
import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Page config
st.set_page_config(
    page_title="GPT Batched Analysis Test",
    page_icon="🧪",
    layout="wide"
)

@st.cache_data
def load_product_database():
    """Load focused 5-product database for shelf testing"""
    return [
        {
            "sku": "Adult_Multivitamin_30s",
            "product": "Youvit Multivitamin Adult",
            "visual": "White pouch, berry images, no mascot"
        },
        {
            "sku": "Collagen_60s",
            "product": "Youvit Collagen",
            "visual": "White pouch, fruit images, pink rose gummy"
        },
        {
            "sku": "Kids_Multivitamin_30s",
            "product": "Youvit Multivitamin Kids",
            "visual": "Blue pouch, cartoon mascot, colorful design"
        },
        {
            "sku": "Kids_Omega3_7s",
            "product": "Youvit Omega3 Kids",
            "visual": "Blue pouch, fish mascot, fish-shaped gummies"
        },
        {
            "sku": "Kids_Vision_7s",
            "product": "Youvit Vision Kids",
            "visual": "Blue pouch, owl mascot, owl-shaped gummies"
        }
    ]

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

def batch_1_shelf_detection(before_img, after_img):
    """Batch 1: Detect numbered shelf positions (vertical: top to bottom)"""
    
    prompt = """
TASK: Number shelves from TOP to BOTTOM in these retail images.

SHELF NUMBERING SYSTEM:
- Shelf 1 = TOP shelf (highest)
- Shelf 2 = Second shelf from top
- Shelf 3 = Third shelf from top  
- Shelf 4 = Bottom shelf (lowest)
- etc.

INSTRUCTIONS:
1. Count shelves from TOP to BOTTOM (vertical positioning only)
2. Identify which shelf has the most Youvit products
3. Horizontal position (left/right) doesn't matter for shelf numbering

Return ONLY this JSON:
{
    "total_shelves": 4,
    "main_shelf": "Shelf 2",
    "all_shelves": ["Shelf 1", "Shelf 2", "Shelf 3", "Shelf 4"],
    "confidence": "HIGH/MEDIUM/LOW"
}
"""
    
    try:
        before_b64 = image_to_base64(before_img)
        after_b64 = image_to_base64(after_img)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "BEFORE:"},
                    {"type": "image_url", "image_url": {"url": before_b64, "detail": "high"}},
                    {"type": "text", "text": "AFTER:"},
                    {"type": "image_url", "image_url": {"url": after_b64, "detail": "high"}}
                ]
            }],
            max_tokens=300,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        
        # Clean JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        st.error(f"❌ Batch 1 Failed: {str(e)}")
        return None

def batch_2_product_recognition(before_img, after_img, shelf_info, product_db):
    """Batch 2: Recognize the 5 core Youvit products on main shelf (no ideal image needed)"""
    
    main_shelf = shelf_info.get("main_shelf", "Shelf 2")
    
    prompt = f"""
TASK: Find these 5 EXPECTED Youvit products on {main_shelf}:

EXPECTED PRODUCTS (IDEAL SETUP):
1. Youvit Multivitamin Adult - White pouch, berry images, no mascot
2. Youvit Collagen - White pouch, fruit images, pink rose gummy
3. Youvit Multivitamin Kids - Blue pouch, cartoon mascot, colorful design
4. Youvit Omega3 Kids - Blue pouch, fish mascot, fish-shaped gummies
5. Youvit Vision Kids - Blue pouch, owl mascot, owl-shaped gummies

INSTRUCTIONS:
1. Look for these specific products on {main_shelf}
2. Count quantities of each product found in BEFORE and AFTER
3. Note position (left, center, right on shelf)
4. Compare what's present in BEFORE vs AFTER

Return ONLY this JSON:
{{
    "before": [
        {{"product": "Youvit Multivitamin Adult", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Collagen", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Multivitamin Kids", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Omega3 Kids", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Vision Kids", "qty": 0, "position": "left/center/right", "found": true/false}}
    ],
    "after": [
        {{"product": "Youvit Multivitamin Adult", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Collagen", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Multivitamin Kids", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Omega3 Kids", "qty": 0, "position": "left/center/right", "found": true/false}},
        {{"product": "Youvit Vision Kids", "qty": 0, "position": "left/center/right", "found": true/false}}
    ],
    "expected_products": ["Youvit Multivitamin Adult", "Youvit Collagen", "Youvit Multivitamin Kids", "Youvit Omega3 Kids", "Youvit Vision Kids"],
    "shelf_analyzed": "{main_shelf}",
    "confidence": "HIGH/MEDIUM/LOW"
}}
"""
    
    try:
        before_b64 = image_to_base64(before_img)
        after_b64 = image_to_base64(after_img)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "BEFORE:"},
                    {"type": "image_url", "image_url": {"url": before_b64, "detail": "high"}},
                    {"type": "text", "text": "AFTER:"},
                    {"type": "image_url", "image_url": {"url": after_b64, "detail": "high"}}
                ]
            }],
            max_tokens=600,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        
        # Clean JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        st.error(f"❌ Batch 2 Failed: {str(e)}")
        return None

def batch_3_before_after_analysis(product_data):
    """Batch 3: Analyze clear changes from before to after vs expected setup"""
    
    expected_products = product_data.get("expected_products", [])
    before_products = [p["product"] for p in product_data.get("before", []) if p.get("found")]
    after_products = [p["product"] for p in product_data.get("after", []) if p.get("found")]
    
    prompt = f"""
TASK: Analyze changes from BEFORE to AFTER compared to EXPECTED setup.

EXPECTED SETUP: {len(expected_products)} products should be present
- Expected: {', '.join(expected_products)}

ACTUAL RESULTS:
- Before: {len(before_products)} products found ({', '.join(before_products) if before_products else 'none'})
- After: {len(after_products)} products found ({', '.join(after_products) if after_products else 'none'})

Return ONLY this JSON:
{{
    "changes_summary": "Clear description of what changed from before to after",
    "products_added": ["products that appeared in after but not in before"],
    "products_removed": ["products that disappeared from before to after"],
    "products_moved": ["products that changed position on shelf"],
    "missing_from_expected": ["expected products still missing in after"],
    "compliance_score": 0-100,
    "improvement_score": 0-100,
    "key_improvements": ["specific improvements made"],
    "remaining_issues": ["what still needs fixing to match expected"],
    "confidence": "HIGH/MEDIUM/LOW"
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=400,
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content
        
        # Clean JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        st.error(f"❌ Batch 3 Failed: {str(e)}")
        return None

def run_batched_analysis(before_img, after_img, product_db):
    """Run sequential batched analysis (no ideal image needed)"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Batch 1: Shelf Detection
    status_text.text("🔍 Batch 1/3: Detecting numbered shelves (top to bottom)...")
    progress_bar.progress(10)
    
    shelf_info = batch_1_shelf_detection(before_img, after_img)
    if not shelf_info:
        st.error("❌ Analysis stopped - Batch 1 failed")
        return None
    
    progress_bar.progress(33)
    main_shelf = shelf_info.get("main_shelf", "Unknown")
    total_shelves = shelf_info.get("total_shelves", 0)
    st.success(f"✅ Batch 1: Found {total_shelves} shelves, analyzing {main_shelf}")
    
    # Batch 2: Product Recognition
    status_text.text(f"📦 Batch 2/3: Finding 5 expected Youvit products on {main_shelf}...")
    progress_bar.progress(40)
    
    product_data = batch_2_product_recognition(before_img, after_img, shelf_info, product_db)
    if not product_data:
        st.error("❌ Analysis stopped - Batch 2 failed")
        return None
    
    progress_bar.progress(66)
    after_found = len([p for p in product_data.get("after", []) if p.get("found")])
    st.success(f"✅ Batch 2: Found {after_found}/5 expected products in AFTER image")
    
    # Batch 3: Before/After Analysis
    status_text.text("📊 Batch 3/3: Analyzing changes BEFORE → AFTER vs expected...")
    progress_bar.progress(80)
    
    analysis_data = batch_3_before_after_analysis(product_data)
    if not analysis_data:
        st.error("❌ Analysis stopped - Batch 3 failed")
        return None
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis completed successfully")
    
    # Combine results
    return {
        "shelf_info": shelf_info,
        "product_data": product_data,
        "analysis": analysis_data,
        "timestamp": datetime.now().isoformat()
    }

# Main App
st.title("🧪 GPT Shelf Analysis Unit Test")
st.caption("Testing 1 shelf with 5 core Youvit products using sequential batch processing")

# Load product database
product_db = load_product_database()
st.info(f"📚 Testing {len(product_db)} core products: Multivitamin Adult, Collagen, Multivitamin Kids, Omega3 Kids, Vision Kids")

# Test requirements
with st.expander("🎯 Test Requirements"):
    st.markdown("""
    **This test analyzes:**
    - **1 main shelf** (numbered top to bottom: Shelf 1=top, Shelf 2=second, etc.)
    - **Vertical positioning only** (horizontal left/right doesn't affect shelf numbering)
    - **5 specific Youvit products** defined in JSON (no ideal image needed)
    - **Clear before→after changes** (what was added, removed, moved)
    - **Compliance with expected setup**
    
    **Expected Products on Main Shelf:**
    1. ✅ Youvit Multivitamin Adult (white pouch, berry images)
    2. ✅ Youvit Collagen (white pouch, fruit images, pink rose)  
    3. ✅ Youvit Multivitamin Kids (blue pouch, cartoon mascot)
    4. ✅ Youvit Omega3 Kids (blue pouch, fish mascot)
    5. ✅ Youvit Vision Kids (blue pouch, owl mascot)
    
    **Shelf Numbering System:**
    - **Shelf 1** = Top shelf (highest)
    - **Shelf 2** = Second from top
    - **Shelf 3** = Third from top  
    - **Shelf 4** = Bottom shelf (lowest)
    """)

# File uploaders (only before and after - no ideal needed)
col1, col2 = st.columns(2)

with col1:
    before_file = st.file_uploader("❌ Before Merchandising", type=['jpg', 'jpeg', 'png'])

with col2:
    after_file = st.file_uploader("✅ After Merchandising", type=['jpg', 'jpeg', 'png'])

# Analysis button
if before_file and after_file:
    
    # Display images
    st.subheader("🖼️ Test Images")
    img_col1, img_col2 = st.columns(2)
    
    before_img = Image.open(before_file)
    after_img = Image.open(after_file)
    
    with img_col1:
        st.image(before_img, caption="Before Merchandising", use_column_width=True)
        
    with img_col2:
        st.image(after_img, caption="After Merchandising", use_column_width=True)
    
    # Run analysis
    if st.button("🚀 Run Batched Analysis", type="primary", use_container_width=True):
        
        results = run_batched_analysis(before_img, after_img, product_db)
        
        if results:
            st.success("🎉 All batches completed successfully!")
            
            # Display results
            st.subheader("📊 Test Results")
            
            # Changes Summary (NEW)
            analysis = results["analysis"]
            st.subheader("📋 Changes Summary (Before → After)")
            
            changes_summary = analysis.get("changes_summary", "No summary available")
            st.info(f"**Summary:** {changes_summary}")
            
            # Changes breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                products_added = analysis.get("products_added", [])
                if products_added:
                    st.success("**✅ Products Added:**")
                    for product in products_added:
                        st.caption(f"• {product}")
                else:
                    st.caption("**✅ Products Added:** None")
                
                products_moved = analysis.get("products_moved", [])
                if products_moved:
                    st.info("**📦 Products Moved:**")
                    for product in products_moved:
                        st.caption(f"• {product}")
                else:
                    st.caption("**📦 Products Moved:** None")
            
            with col2:
                products_removed = analysis.get("products_removed", [])
                if products_removed:
                    st.error("**❌ Products Removed:**")
                    for product in products_removed:
                        st.caption(f"• {product}")
                else:
                    st.caption("**❌ Products Removed:** None")
                
                missing_from_expected = analysis.get("missing_from_expected", [])
                if missing_from_expected:
                    st.warning("**⚠️ Still Missing from Expected:**")
                    for product in missing_from_expected:
                        st.caption(f"• {product}")
                else:
                    st.caption("**⚠️ Missing from Expected:** None")
            
            # Metrics
            st.subheader("📈 Scores")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Compliance Score", f"{analysis.get('compliance_score', 0)}%")
            
            with col2:
                st.metric("Improvement Score", f"{analysis.get('improvement_score', 0)}%")
            
            with col3:
                shelf_analyzed = results["product_data"].get("shelf_analyzed", "Unknown")
                st.metric("Shelf Analyzed", shelf_analyzed)
            
            # Product detection details
            st.subheader("🔍 5-Product Detection Results")
            
            product_data = results["product_data"]
            
            # Create tabs for before and after only
            tab1, tab2 = st.tabs(["❌ Before", "✅ After"])
            
            phases = [
                (tab1, "before", "Before"), 
                (tab2, "after", "After")
            ]
            
            for tab, phase_key, phase_name in phases:
                with tab:
                    phase_products = product_data.get(phase_key, [])
                    found_count = len([p for p in phase_products if p.get("found")])
                    
                    st.write(f"**{phase_name}:** {found_count}/5 expected products found")
                    
                    for product in phase_products:
                        product_name = product.get("product", "Unknown")
                        found = product.get("found", False)
                        qty = product.get("qty", 0)
                        position = product.get("position", "unknown")
                        
                        if found:
                            st.success(f"✅ {product_name} - Qty: {qty} - Position: {position}")
                        else:
                            st.error(f"❌ {product_name} - Not found")
            
            # Expected vs Actual comparison
            st.subheader("🎯 Expected vs Actual")
            expected_products = product_data.get("expected_products", [])
            
            exp_col1, exp_col2 = st.columns(2)
            
            with exp_col1:
                st.write("**Expected Products (5):**")
                for product in expected_products:
                    st.caption(f"✅ {product}")
            
            with exp_col2:
                after_products = product_data.get("after", [])
                found_products = [p["product"] for p in after_products if p.get("found")]
                st.write(f"**Actually Found ({len(found_products)}):**")
                for product in found_products:
                    st.caption(f"✅ {product}")
                
                if len(found_products) < 5:
                    missing = len(expected_products) - len(found_products)
                    st.caption(f"❌ {missing} products still missing")
            
            # Shelf detection info
            st.subheader("🏪 Shelf Detection")
            
            shelf_info = results["shelf_info"]
            shelf_col1, shelf_col2 = st.columns(2)
            
            with shelf_col1:
                total_shelves = shelf_info.get("total_shelves", 0)
                all_shelves = shelf_info.get("all_shelves", [])
                st.write(f"**Total Shelves:** {total_shelves}")
                st.write(f"**All Shelves:** {', '.join(all_shelves)}")
            
            with shelf_col2:
                main_shelf = shelf_info.get("main_shelf", "Unknown")
                confidence = shelf_info.get("confidence", "MEDIUM")
                st.write(f"**Main Shelf Analyzed:** {main_shelf}")
                st.write(f"**Detection Confidence:** {confidence}")
            
            # Improvements and Issues
            st.subheader("💡 Insights")
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                improvements = analysis.get("key_improvements", [])
                if improvements:
                    st.success("**✅ Key Improvements:**")
                    for improvement in improvements:
                        st.caption(f"• {improvement}")
                else:
                    st.caption("**✅ Key Improvements:** None identified")
            
            with insight_col2:
                issues = analysis.get("remaining_issues", [])
                if issues:
                    st.warning("**⚠️ Remaining Issues:**")
                    for issue in issues:
                        st.caption(f"• {issue}")
                else:
                    st.caption("**⚠️ Remaining Issues:** None")
            
            # Raw results for debugging
            with st.expander("🔧 Raw Results (Debug)"):
                st.json(results)
            
            # Export results
            st.download_button(
                label="📥 Download Test Results",
                data=json.dumps(results, indent=2),
                file_name=f"5product_shelf_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

else:
    st.info("👆 Upload BEFORE and AFTER images to start testing")
    st.caption("💡 No ideal image needed - expected setup is defined in the product database")

# Footer
st.markdown("---")
st.caption("🧪 Unit Test v2.4 - No Ideal Image Required")
st.caption("🎯 Testing: Vertical shelf numbering (top→bottom) + JSON-defined expected products + Before→After analysis")