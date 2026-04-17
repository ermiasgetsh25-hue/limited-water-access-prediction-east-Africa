import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Set page config FIRST (must be first Streamlit command)
st.set_page_config(
    page_title="Water Access Prediction - East Africa",
    page_icon="💧",
    layout="wide"
)

# Load model with no loading message
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(r"C:\Users\wub\Desktop\peda\best_xgboost_model.pkl")

model = load_model()

# ============================================
# YOUDEN THRESHOLD (calculated from your test data)
# ============================================
YOUden_THRESHOLD = 0.4655

# ============================================
# PAGE CONTENT
# ============================================
st.title("💧 Machine Learning Prediction of Household Limited Access to Improved Water Services")
st.markdown("### East Africa Regional Prediction Tool")
st.markdown("---")

# ============================================
# SIDEBAR - MODEL INFORMATION
# ============================================
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    **Algorithm:** XGBoost  
    **Region:** East Africa  
    **Outcome:** Limited vs Basic Water Access  
    **Features:** 32 household & community variables  
    """)
    
    st.metric("Classification Threshold", f"{YOUden_THRESHOLD:.3f}")
    st.caption("Optimal Youden Index threshold")
    
    st.divider()
    st.markdown("**Performance Metrics (Test Set)**")
    st.markdown("""
    - Accuracy: 0.7582
    - Precision: 0.8237
    - Recall: 0.7582
    - F1-Score: 0.7758
    - ROC-AUC: 0.853
    """)
    
    st.divider()
    st.caption("📝 **Citation:** [Your Paper Citation]")
    st.caption("⚠️ **Disclaimer:** Decision support tool only. Field validation recommended.")

# ============================================
# MAIN INPUT FORM
# ============================================
st.header("📋 Enter Household Information")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Household Demographics")
    
    # Country
    country = st.selectbox(
        "Country",
        ["Burundi", "Ethiopia", "Kenya", "Madagascar", "Malawi", 
         "Mozambique", "Rwanda", "Tanzania", "Uganda", "Zambia", "Zimbabwe"]
    )
    
    # Type of place of residence
    residence = st.radio("Type of Place of Residence", ["Rural", "Urban"], horizontal=True)
    
    # Sex of head of household
    sex_head = st.radio("Sex of Household Head", ["male", "female"], horizontal=True)
    
    # Current marital status
    marital_status = st.selectbox(
        "Current Marital Status of Head",
        ["Never married", "Married", "Divorced", "Widowed"]
    )
    
    # Highest educational level
    education = st.selectbox(
        "Highest Educational Level Attained",
        ["No education", "Primary", "Secondary", "Higher"]
    )
    
    # Wealth index
    wealth = st.selectbox(
        "Wealth Index Combined",
        ["Poorest", "Poorer", "Middle", "Richer", "Richest"]
    )
    
    # Household size
    hh_size = st.radio("Number of Household Members", ["<4", ">=4"], horizontal=True)
    
    # Children under 5 group
    children_u5 = st.selectbox(
        "Children Under 5 Group",
        ["no child", "1-2", ">=3"]
    )
    
    # Head age group
    head_age = st.selectbox(
        "Household Head Age Group",
        ["<35", "35-60", ">60"]
    )

with col2:
    st.subheader("🏡 Household Facilities & Services")
    
    # Electricity
    electricity = st.radio("Has Electricity", ["no", "yes"], horizontal=True)
    
    # Water treatment
    water_treatment = st.radio("Household Water Treatment", ["no", "yes"], horizontal=True)
    
    # Media exposure
    media_exposure = st.radio("Household Media Exposure", ["No", "Yes"], horizontal=True)
    
    # Sanitation status
    sanitation = st.radio(
        "Sanitation Status",
        ["unimproved_sanitation", "improved_sanitation"],
        horizontal=True
    )
    
    # Handwashing place observed
    handwashing = st.radio("Handwashing Place Observed", ["no", "yes"], horizontal=True)
    
    # Share toilet
    share_toilet = st.radio("Share Toilet with Other Households", ["no", "yes"], horizontal=True)
    
    # Soap/detergent present
    soap = st.radio("Items Present: Soap or Detergent", ["no", "yes"], horizontal=True)
    
    # Housing material status
    housing_material = st.selectbox(
        "Housing Material Status",
        ["unimproved_housing_material", "improved_housing_material"]
    )
    
    # Person fetching water
    person_fetching = st.selectbox(
        "Person Fetching Water",
        ["Adult man", "Adult woman", "female_child_under_15", "male_child_under_15", "others"]
    )

# ============================================
# COMMUNITY LEVEL VARIABLES
# ============================================
st.subheader("🌍 Community-Level Variables")
st.caption("These variables represent the community average/percentage")

col3, col4 = st.columns(2)

with col3:
    community_electricity = st.selectbox("Community Level Electricity", ["Low", "High"])
    community_water_treatment = st.selectbox("Community Level Water Treatment", ["Low", "High"])
    community_media = st.selectbox("Community Level Media Exposure", ["Low", "High"])

with col4:
    community_poverty = st.selectbox("Community Level Poverty", ["Low", "High"])
    community_education = st.selectbox("Community Level Education", ["Low", "High"])
    community_toilet = st.selectbox("Community Level Toilet Facility", ["Low", "High"])

# ============================================
# ENCODING FUNCTION - MATCHES MODEL EXPECTATIONS
# ============================================
def encode_features():
    # Binary features (0/1)
    residence_encoded = 0 if residence == "Rural" else 1
    electricity_encoded = 0 if electricity == "no" else 1
    sex_head_encoded = 0 if sex_head == "female" else 1
    water_treatment_encoded = 0 if water_treatment == "no" else 1
    media_exposure_encoded = 0 if media_exposure == "No" else 1
    sanitation_encoded = 0 if sanitation == "unimproved_sanitation" else 1
    handwashing_encoded = 0 if handwashing == "no" else 1
    share_toilet_encoded = 0 if share_toilet == "no" else 1
    soap_encoded = 0 if soap == "no" else 1
    hh_size_encoded = 0 if hh_size == "<4" else 1
    housing_material_encoded = 0 if housing_material == "unimproved_housing_material" else 1
    
    # Multi-class encodings
    education_map = {"No education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
    education_encoded = education_map[education]
    
    wealth_map = {"Poorest": 0, "Poorer": 1, "Middle": 2, "Richer": 3, "Richest": 4}
    wealth_encoded = wealth_map[wealth]
    
    # Community variables (Low=0, High=1)
    community_electricity_encoded = 0 if community_electricity == "Low" else 1
    community_water_treatment_encoded = 0 if community_water_treatment == "Low" else 1
    community_media_encoded = 0 if community_media == "Low" else 1
    community_poverty_encoded = 0 if community_poverty == "Low" else 1
    community_education_encoded = 0 if community_education == "Low" else 1
    community_toilet_encoded = 0 if community_toilet == "Low" else 1
    
    # Children under 5 encoding - create binary columns
    children_u5_no_child = 1 if children_u5 == "no child" else 0
    children_u5_one_to_two = 1 if children_u5 == "1-2" else 0
    # Note: ">=3" is reference category (both columns = 0)
    
    # Head age group encoding - binary columns matching model
    head_age_35 = 1 if head_age == "35-60" else 0
    head_age_60 = 1 if head_age == ">60" else 0
    # Note: "<35" is reference category (both = 0)
    
    # Marital status encoding - binary columns
    marital_married = 1 if marital_status == "Married" else 0
    marital_widowed = 1 if marital_status == "Widowed" else 0
    # Note: "Never married" and "Divorced" are reference categories
    
    # Person fetching water encoding - binary columns
    person_male_child = 1 if person_fetching == "male_child_under_15" else 0
    person_female_child = 1 if person_fetching == "female_child_under_15" else 0
    person_adult_woman = 1 if person_fetching == "Adult woman" else 0
    # Note: "Adult man" and "others" are reference categories
    
    # Country encoding - binary columns (one-hot encoding)
    country_rwanda = 1 if country == "Rwanda" else 0
    country_malawi = 1 if country == "Malawi" else 0
    country_uganda = 1 if country == "Uganda" else 0
    country_madagascar = 1 if country == "Madagascar" else 0
    # Note: All other countries (Burundi, Ethiopia, Kenya, Mozambique, Tanzania, Zambia, Zimbabwe)
    # are reference categories (all 4 columns = 0)
    
    # Create feature dictionary with EXACT column names model expects
    features = {
        'Wealth index combined': wealth_encoded,
        'highest_educational_level_attained': education_encoded,
        'share_toilet_with_other': share_toilet_encoded,
        'household_media_exposure': media_exposure_encoded,
        'hh_head_age_group_35': head_age_35,
        'Number of households Members': hh_size_encoded,
        'household_sanitation_status': sanitation_encoded,
        'handwashing_place_observed': handwashing_encoded,
        'community_level_toilet': community_toilet_encoded,
        'sex_of_head_of_household': sex_head_encoded,
        'household_water_treatment': water_treatment_encoded,
        'community_level_education': community_education_encoded,
        'community_level_poverty': community_poverty_encoded,
        'hh_head_age_group_60': head_age_60,
        'community_level_electricity': community_electricity_encoded,
        'housing_material_status': housing_material_encoded,
        'items_present_soap_or_detergent': soap_encoded,
        'community_level_media_exposure': community_media_encoded,
        'children_under5_group_one_to_two': children_u5_one_to_two,
        'community_level_water_treatment ': community_water_treatment_encoded,  # Note: space at end
        'has_electricity': electricity_encoded,
        'type_of_place_of_residence': residence_encoded,
        'children_under5_group_no child': children_u5_no_child,
        'current_marital_status_Married': marital_married,
        'person_fetching_water_male_child_U15': person_male_child,
        'country_Rwanda': country_rwanda,
        'person_fetching_water_female_child_U15': person_female_child,
        'current_marital_status_Widowed': marital_widowed,
        'person_fetching_water_adult_woman': person_adult_woman,
        'country_Malawi': country_malawi,
        'country_Uganda': country_uganda,
        'country_Madagascar': country_madagascar,
    }
    
    return features

# ============================================
# PREDICTION BUTTON AND OUTPUT
# ============================================
st.markdown("---")
if st.button("🔍 Predict Water Access Status", type="primary", use_container_width=True):
    
    with st.spinner("Analyzing household characteristics..."):
        # Get encoded features
        features_dict = encode_features()
        
        # Convert to DataFrame (single row)
        features_df = pd.DataFrame([features_dict])
        
        # Get prediction probability (probability of Limited Access = class 1)
        probability = model.predict_proba(features_df)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Create metrics row
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.metric("Predicted Probability", f"{probability:.2%}")
        
        with col_result2:
            risk_level = "HIGH RISK" if probability >= YOUden_THRESHOLD else "LOW RISK"
            st.metric("Risk Level", risk_level)
        
        with col_result3:
            st.metric("Water Access Classification", 
                     "Limited Access" if probability >= YOUden_THRESHOLD else "Basic Access")
        
        # Detailed output with recommendations
        if probability >= YOUden_THRESHOLD:
            st.error("⚠️ **HIGH RISK: Household predicted to have LIMITED access to improved water services**")
            st.warning("""
            **Recommendations:**
            - Prioritize for water infrastructure intervention
            - Assess existing water sources for improvement
            - Consider community-level water supply programs
            - Provide household water treatment support
            """)
        else:
            st.success("✅ **LOW RISK: Household predicted to have BASIC access to improved water services**")
            st.info("""
            **Recommendations:**
            - Maintain current water access
            - Monitor for potential deterioration
            - Community education on water quality maintenance
            """)
        
        # Show threshold information
        st.caption(f"*Classification based on Youden Index threshold: {YOUden_THRESHOLD:.3f}*")
        
        # View Input Summary
        with st.expander("📋 View Input Summary"):
            col_sum1, col_sum2 = st.columns(2)
            with col_sum1:
                st.markdown("**Household Characteristics**")
                st.write(f"- Country: {country}")
                st.write(f"- Residence: {residence}")
                st.write(f"- Household size: {hh_size}")
                st.write(f"- Wealth quintile: {wealth}")
                st.write(f"- Education: {education}")
                st.write(f"- Electricity: {electricity}")
                st.write(f"- Sanitation: {sanitation}")
            
            with col_sum2:
                st.markdown("**Community Context**")
                st.write(f"- Community electricity level: {community_electricity}")
                st.write(f"- Community poverty level: {community_poverty}")
                st.write(f"- Community education level: {community_education}")
                st.write(f"- Water treatment community level: {community_water_treatment}")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("© 2024 | Machine Learning for Water Access Prediction | East Africa Regional Tool")