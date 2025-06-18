import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor, StackingRegressor

# -----------------------------
# 1. Load Data (Synthetic or Generate New)
# -----------------------------
original_data_file = "QuestionsData.csv"
synthetic_data_file = "SyntheticData.csv"

if os.path.exists(synthetic_data_file):
    st.write("✅ Found existing synthetic data. Loading...")
    data_df = pd.read_csv(synthetic_data_file, sep=',')
else:
    st.write("🔄 Generating synthetic data...")
    data_df = pd.read_csv(original_data_file, sep=',', dtype=str)
    data_df.columns = data_df.columns.str.strip()
    
    data_df = data_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    numeric_columns = ['Experience_Years', 'Games_NC', 'Games_LC', 'Games_TC',
                       'NC_Challenge', 'LC_Challenge', 'TC_Challenge',
                       'Cost_NC', 'Cost_LC', 'Cost_TC',
                       'Time_NC', 'Time_LC', 'Time_TC', 'Skill_Level']
    for col in numeric_columns:
        if col in data_df.columns:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    data_df.replace("null", np.nan, inplace=True)

    def generate_synthetic_data(df, target_rows=1_000):
        original_rows = df.copy()
        original_count = len(original_rows)

        multiplier = max(1, (target_rows - original_count) // original_count + 1)
        synthetic_part = pd.concat([original_rows] * multiplier, ignore_index=True)
        synthetic_part = synthetic_part.sample(n=target_rows - original_count, random_state=42).reset_index(drop=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        noise = np.random.normal(0, 1, (synthetic_part.shape[0], len(numeric_cols)))
        synthetic_part[numeric_cols] = np.round(synthetic_part[numeric_cols] + noise)

        for col in ['Time_NC', 'Time_LC', 'Time_TC', 'NC_Challenge', 'LC_Challenge', 'TC_Challenge', 
                    'Skill_Level', 'Cost_NC', 'Cost_LC', 'Cost_TC']:
            if col in synthetic_part.columns:
                synthetic_part[col] = synthetic_part[col].clip(0, 5)

        for col in ['Experience_Years', 'Games_NC', 'Games_LC', 'Games_TC']:
            if col in synthetic_part.columns:
                synthetic_part[col] = synthetic_part[col].clip(0).astype(int)

        # Assign new Developer_IDs to synthetic part, starting after the max from original
        max_id = original_rows['Developer_ID'].astype(int).max() if 'Developer_ID' in original_rows.columns else 0
        synthetic_part['Developer_ID'] = range(max_id + 1, max_id + 1 + len(synthetic_part))

        # Combine original and synthetic
        full_df = pd.concat([original_rows, synthetic_part], ignore_index=True)

        return full_df

    data_df = generate_synthetic_data(data_df, target_rows=50_000)
    # 🔹 Outlier Removal based on statistical analysis
    def remove_outliers_iqr(df, columns):
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        return df

    outlier_sensitive_columns = [
        'Experience_Years', 'Time_NC', 'Time_LC', 'Time_TC', 
        'Cost_NC', 'Cost_LC', 'Cost_TC', 
        'NC_Challenge', 'LC_Challenge', 'TC_Challenge'
    ]

    # Apply outlier filtering
    data_df = remove_outliers_iqr(data_df, outlier_sensitive_columns)
    data_df.to_csv(synthetic_data_file, index=False)
    st.write("✅ Synthetic data generated and saved.")

# -----------------------------
# 2. Data Cleaning (with Scaling to preserve original range 1-5)
# -----------------------------
def clean_data(df):
    bool_columns = ['Proficiency_NC', 'Proficiency_LC', 'Proficiency_TC']
    numeric_columns = ['Games_NC', 'Games_LC', 'Games_TC', 'NC_Challenge', 'LC_Challenge', 'TC_Challenge',
                       'Cost_NC', 'Cost_LC', 'Cost_TC', 'Time_NC', 'Time_LC', 'Time_TC']

    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Apply MinMax scaling but maintain the original scale (1-5)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

data_df = clean_data(data_df)

# -----------------------------
# 3. Encode Techniques as Binary Variables per Category
# -----------------------------
techniques_NC = ["Drag-and-Drop", "Visual-Scripting", "Template-Based", "Event-Based", "Asset-Marketplace", "Code-Free-Animation"]
techniques_LC = ["Dynamic-Scripting", "Visual/Textual-Scripting", "Code-Blocks", "API-Integration", "Templates-Plugins"]
techniques_TC = ["Object-Oriented", "Custom-Game-Engines", "Physics/Graphics-Programming", "Writing-AI-Scripts", "Backend/Server-Development", "Procedural-Content-Generation", "Custom-Animations/Assets"]

def encode_techniques(df, col_name, techniques_list, prefix):
    for tech in techniques_list:
        new_col = f"{prefix}_{tech}"
        df[new_col] = df[col_name].apply(lambda x: 1 if isinstance(x, str) and tech in x else 0)
    return df

if "Techniques_NC" in data_df.columns:
    data_df = encode_techniques(data_df, "Techniques_NC", techniques_NC, "NC")
if "Techniques_LC" in data_df.columns:
    data_df = encode_techniques(data_df, "Techniques_LC", techniques_LC, "LC")
if "Techniques_TC" in data_df.columns:
    data_df = encode_techniques(data_df, "Techniques_TC", techniques_TC, "TC")

# -----------------------------
# 4. Streamlit App: Layout, Visualization, and Model Training
# -----------------------------
st.title("Χ-CoΔ")

# Tab Navigation
tabs = ["Application", "Statistical Analysis", "README"]
active_tab = st.selectbox("Select Tab", tabs, key="main_tab")

# --- APPLICATION TAB ---
def application():
    st.subheader("🛠 Cost, Time & Challenge Prediction Model")

    st.info("NC: No-Code game development platforms, " \
    "LC: No-Code game development platforms and " \
    "TC: Traditional-Coding game development platforms.")
    
    # Layout: Create three columns for user input
    col1, col2, col3 = st.columns(3)
    
    # Column 1: Select Platform Type
    with col1:
        platform = st.selectbox("Select Platform Type", ['NC', 'LC', 'TC'], key="platform_select")
    
    # Column 2: Techniques Multiselect based on Platform
    with col2:
        if platform == 'NC':
            technique_options = techniques_NC
        elif platform == 'LC':
            technique_options = techniques_LC
        else:
            technique_options = techniques_TC
        selected_techniques = st.multiselect("Select Techniques", technique_options, key="techniques_select")
    
    # Column 3: Select Regression Algorithm (Added new models)
    with col3:
        algorithm_choice = st.selectbox("Select Regression Algorithm", [
            "Random Forest Regression",
            "Gradient Boosting Regression",
            "Linear Regression",
            "Decision Tree Regression",
            "Extra Trees Forest",
            "K-Neighbors Regression",
            "Support Vector Regression",
            "XGBoost Regression",
            "MLP Neural Network",
            "Bayesian Ridge Regression",
            "Lasso Regression",
            "CatBoost Regression",
            "Gaussian Process Regression",
            "AdaBoost Regression",
            "Stacking Ensemble"
        ], key="algorithm_select")
    
    st.markdown("---")

    # New inputs: Games created, Experience, Skill Level
    st.markdown("### 👤 User Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        games_created = st.slider("Games Created (0–10+)", 0, 10, 0)
    with col5:
        experience_years = st.slider("Experience in Years (0–10+)", 0, 10, 0)
    with col6:
        skill_level = st.slider("Skill Level (0–5)", 0, 5, 0)
    
    st.markdown("---")
    
    # Data Distribution Plot
    st.info("The graph below is the normalized version of the Cost data for the category you have selected, showcasing how much is the average cost of using the specific category based on the data collected. " \
    "Please keep in mind that 0 represents the null value from the people that never used this kind of platform before. Therefore, anything between 0.1 - 1 is a valid value.")
    st.subheader("📈 Data Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    cost_col = f'Cost_{platform}'
    if cost_col in data_df.columns:
        sns.histplot(data_df[cost_col].dropna(), bins=30, kde=True, ax=ax, color="blue")
        ax.set_title(f"Distribution of {cost_col}")
        st.pyplot(fig)
    else:
        st.error(f"Column {cost_col} not found in dataset.")
    
    st.warning("Warning: Depending on which techniques and how many of them you have selected, the results of the training model you chose will vary. "
            "General performance may differ per category. For example, tree-based models (Random Forest, Extra Trees, etc.) "
            "tend to perform robustly with many binary features, while methods like Linear Regression or Lasso might underperform "
            "if the underlying relationships are nonlinear.")

    if st.button("Estimate", key="estimate_button"):
        if not selected_techniques:
            st.error("Please select at least one technique before estimating.")
        else:
            df_filtered = data_df.copy()

            tech_cols = [f"{platform}_{tech}" for tech in selected_techniques if f"{platform}_{tech}" in df_filtered.columns]
            if tech_cols:
                df_filtered = df_filtered[df_filtered[tech_cols].sum(axis=1) >= 1]
            else:
                st.error("No valid technique columns found for the selected platform.")
                return

            # Map platform to the correct 'Games_' column
            games_col = f"Games_{platform}"

            # Filter rows with non-null targets and fill missing data
            targets = {
                "Cost": f"Cost_{platform}",
                "Time": f"Time_{platform}",
                "Challenge": f"{platform}_Challenge"
            }

            results = {}
            features = tech_cols + [games_col, 'Experience_Years', 'Skill_Level']

            # Prepare X and y
            if games_col not in df_filtered.columns:
                st.error(f"Column {games_col} not found in data.")
                return

            df_filtered['Experience_Years'] = experience_years
            df_filtered['Skill_Level'] = skill_level
            df_filtered[games_col] = games_created
            
            for key, target_col in targets.items():
                if target_col not in df_filtered.columns:
                    st.error(f"Target column {target_col} not found in dataset.")
                    continue

                y = df_filtered[target_col].fillna(df_filtered[target_col].median())
                X = df_filtered[features].fillna(0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Choose algorithm
                if algorithm_choice == "Random Forest Regression":
                    model = RandomForestRegressor()
                elif algorithm_choice == "Gradient Boosting Regression":
                    model = GradientBoostingRegressor()
                elif algorithm_choice == "Linear Regression":
                    model = LinearRegression()
                elif algorithm_choice == "Decision Tree Regression":
                    model = DecisionTreeRegressor()
                elif algorithm_choice == "Extra Trees Forest":
                    model = ExtraTreesRegressor()
                elif algorithm_choice == "K-Neighbors Regression":
                    model = KNeighborsRegressor()
                elif algorithm_choice == "Support Vector Regression":
                    model = SVR()
                elif algorithm_choice == "XGBoost Regression":
                    model = XGBRegressor()
                elif algorithm_choice == "MLP Neural Network":
                    model = MLPRegressor()
                elif algorithm_choice == "Bayesian Ridge Regression":
                    model = BayesianRidge()
                elif algorithm_choice == "Lasso Regression":
                    model = Lasso()
                elif algorithm_choice == "CatBoost Regression":
                    model = CatBoostRegressor(verbose=0)
                elif algorithm_choice == "Gaussian Process Regression":
                    model = GaussianProcessRegressor()
                elif algorithm_choice == "AdaBoost Regression":
                    model = AdaBoostRegressor()
                elif algorithm_choice == "Stacking Ensemble":
                    estimators = [
                        ('lr', LinearRegression()),
                        ('rf', RandomForestRegressor())
                    ]
                    model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
                else:
                    st.error("Unknown algorithm selected.")
                    continue

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                median_pred = np.median(y_pred)
                estimated_value = median_pred * 4 + 1

                results[key] = {
                    "Estimated": estimated_value,
                    "MSE": mse,
                    "R2": r2
                }


            st.subheader("📊 Estimation Results")
            for metric, res in results.items():
                st.write(f"**Estimated {metric}:** {res['Estimated']:.2f} / 5")
                st.write(f"**Mean Squared Error:** {res['MSE']:.2f}")
                if res['MSE'] < 0.1:
                    st.markdown("<span style='color: green; font-size: 12px;'>Good: Lower MSE means better model performance.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: red; font-size: 12px;'>Bad: Higher MSE indicates the model is making large errors.</span>", unsafe_allow_html=True)
                st.write(f"**R-squared Score:** {res['R2']:.2f}")
                if res['R2'] > 0.7:
                    st.markdown("<span style='color: green; font-size: 12px;'>Good: Model explains most of the variance.</span>", unsafe_allow_html=True)
                elif res['R2'] > 0.3:
                    st.markdown("<span style='color: yellow; font-size: 12px;'>Moderate: The model has some predictive power.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color: red; font-size: 12px;'>Bad: The model is not capturing patterns well.</span>", unsafe_allow_html=True)
                st.markdown("---")

def statistical_analysis():
    st.title("Statistical Analysis of Questionnaire")

    # File selection: original, synthetic, or uploaded
    data_option = st.radio(
        "Select Data:",
        ("Questionnaire Data", "Synthetic Data", "Upload CSV"),
        horizontal=True
    )

    # File paths
    original_data_file = "QuestionsData.csv"
    synthetic_data_file = "SyntheticData.csv"

    # Load selected data
    if data_option == "Questionnaire Data":
        df = pd.read_csv(original_data_file)
    elif data_option == "Synthetic Data":
        df = pd.read_csv(synthetic_data_file)
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload a CSV file to proceed.")
            return  # Stop execution if no file is uploaded

    # Download option for the selected dataset
    st.download_button(
        label="Download Current Dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="selected_data.csv",
        mime="text/csv"
    )

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    # --- Shared Setup for Statistical Analysis ---

  # Helper: Get category from either prefix or suffix
    def get_category(col_name):
        for label in category_labels:
            if col_name.startswith(f"{label}_") or col_name.endswith(f"_{label}"):
                return label
        return None
    
    def get_category_prefix(col_name):
        if "Challenge" in col_name:
            return "Challenge"
        elif "Cost" in col_name:
            return "Cost"
        elif "Time" in col_name:
            return "Time"
        elif "Skill" in col_name:
            return "Skill"
        elif "Platform" in col_name:
            return "Platform"
        elif "Beginner" in col_name:
            return "Beginner"
        else:
            return None

    # Group techniques (assumes techniques_NC, techniques_LC, etc. are defined earlier)
    NC_techs = [f"NC_{tech}" for tech in techniques_NC]
    LC_techs = [f"LC_{tech}" for tech in techniques_LC]
    TC_techs = [f"TC_{tech}" for tech in techniques_TC]

    # Define the known category labels
    category_labels = ["NC", "LC", "TC"]

    def get_tech_group(col):
        if col in NC_techs:
            return "NC"
        elif col in LC_techs:
            return "LC"
        elif col in TC_techs:
            return "TC"
        return None

    # Define core groups
    challenge_cols = ["NC_Challenge", "LC_Challenge", "TC_Challenge"]
    cost_cols = [col for col in data_df.columns if col.startswith("Cost_")]
    time_cols = [col for col in data_df.columns if col.startswith("Time_")]

    def is_same_group(col1, col2, group):
        return col1 in group and col2 in group

    # Section: Descriptive Statistics
    st.header("📊 Descriptive Statistics")
    st.dataframe(df[numerical_cols].describe().transpose())

    # Section: Distribution Plots
    st.header("📈 Distribution")
    for col in numerical_cols:
        if col == "Developer_ID":
            continue  # Skip this column
        data = df[col].dropna()
        fig, ax = plt.subplots()
        # Set bins for whole numbers (ensure full range is covered)
        min_val, max_val = int(data.min()), int(data.max())
        bins = range(min_val, max_val + 2)  # +2 to include the last bin fully
        sns.histplot(data, bins=bins, kde=False, ax=ax, discrete=True)
        ax.set_title(f"Distribution for {col}")
        ax.set_xticks(range(min_val, max_val + 1))  # Make sure all ticks show
        st.pyplot(fig)

    # Section: Pearson Correlation Matrix (All numerical + encoded techniques)
    st.subheader("🔗 Pearson Correlation Matrix (Full Dataset)")

    # Collect all numeric columns and encoded technique columns
    numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
    all_prefixed_cols = [col for col in data_df.columns if col.startswith(('NC_', 'LC_', 'TC_'))]
    correlation_cols = list(set(numeric_cols + all_prefixed_cols))

    # Compute correlation matrix
    corr_matrix = data_df[correlation_cols].corr()

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(25, 20))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", square=True, linewidths=0.8, cbar_kws={'shrink': 0.9}, ax=ax)
    ax.set_title("Full Pearson Correlation Matrix", fontsize=18)
    st.pyplot(fig)

    # Updated thresholds and label-color mapping
    # Correlation Coefficients: Appropriate Use and Interpretation Patrick Schober, MD, PhD, MMedStat, Christa Boer, PhD, MSc, and Lothar A. Schwarte, MD, PhD, MBA
    def categorize_correlation(abs_val):
        if abs_val >= 0.90:
            return "Very strong correlation", "darkred"
        elif abs_val >= 0.60:
            return "Strong correlation", "red"
        elif abs_val >= 0.25:
            return "Moderate correlation", "orange"
        elif abs_val >= 0.10:
            return "Weak correlation", "blue"
        else:
            return "Negligible correlation", "gray"

    # Optionally: Let user select and inspect strong correlations
    st.subheader("📌 Significant Correlation Insights (|corr| ≥ 0.1)")
    display_threshold = 0.10  # Show weak and higher

    # Detect encoded techniques more accurately
    technique_cols = [
        col for col in correlation_cols
        if any(col.startswith(f"{prefix}_") for prefix in category_labels)
        and "_" in col
        and not col.endswith("Challenge")
        and not col.startswith(("Cost_", "Time_"))
    ]

    # Flatten upper triangle of the matrix to list pairs with filtering
    significant_pairs = []
    for i in range(len(correlation_cols)):
        for j in range(i + 1, len(correlation_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= display_threshold:
                col1 = correlation_cols[i]
                col2 = correlation_cols[j]

                if is_same_group(col1, col2, cost_cols + time_cols + challenge_cols + technique_cols):
                    continue

                cat1 = get_category(col1)
                cat2 = get_category(col2)
                if cat1 and cat2 and cat1 != cat2:
                    continue

                significant_pairs.append((col1, col2, corr_val))

    # Sort by correlation magnitude
    significant_pairs = sorted(significant_pairs, key=lambda x: abs(x[2]), reverse=True)

    # Display results with colored strength tags
    if significant_pairs:
        for col1, col2, val in significant_pairs:
            direction = "positive" if val > 0 else "negative"
            strength, color = categorize_correlation(abs(val))
            st.markdown(
                f"""- <b>{col1}</b> vs <b>{col2}</b> → 
                <code>Correlation = {val:.2f}</code> 
                (<i>{direction},</i> <span style="color:{color}; font-weight:bold;">{strength}</span>)""",
                unsafe_allow_html=True
            )
    else:
        st.info(f"No significant correlations found with threshold ≥ {display_threshold}")

    # Define thresholds for p-value strength interpretation
    def categorize_p_value(p_val):
        if p_val < 0.001:
            return "Very Strong"
        elif p_val < 0.01:
            return "Strong"
        elif p_val < 0.05:
            return "Moderate"
        else:
            return "Not Significant"

    # Extract category from column name (either as prefix or suffix)
    def get_category(col_name):
        for label in ["NC", "LC", "TC"]:
            if col_name.startswith(f"{label}_") or col_name.endswith(f"_{label}"):
                return label
        return None

    chi2_results = []

    for i, col1 in enumerate(categorical_cols):
        for j, col2 in enumerate(categorical_cols):
            if j <= i:
                continue  # Avoid duplicate pairs

            # Skip low-variability columns
            if df[col1].nunique() <= 1 or df[col2].nunique() <= 1:
                continue

            prefix1 = get_category(col1)
            prefix2 = get_category(col2)

            if is_same_group(col1, col2, cost_cols + time_cols + challenge_cols + technique_cols):
                continue

            if col1 in technique_cols and col2 in technique_cols:
                group1 = get_tech_group(col1)
                group2 = get_tech_group(col2)
                if group1 != group2:
                    continue

            if prefix1 and prefix2 and prefix1 != prefix2:
                continue

            contingency_table = pd.crosstab(df[col1], df[col2])

            try:
                chi2, p, dof, expected = chi2_contingency(contingency_table)

                # Check expected frequencies
                if (expected < 5).sum() > 0.2 * expected.size or (expected < 1).any():
                    continue

                if p < 0.05:
                    strength = categorize_p_value(p)
                    chi2_results.append((col1, col2, p, strength, chi2, dof))
            except ValueError:
                continue

    # Optional: Color mapping
    def get_strength_color(strength):
        return {
            "Very Strong": "#d62728",      # red
            "Strong": "#ff7f0e",           # orange
            "Moderate": "#2ca02c",         # green
            "Not Significant": "#1f77b4"   # blue
        }.get(strength, "#000000")         # fallback: black

    # Display results
    if chi2_results:
        st.subheader("📌 Significant Chi-Square Associations (p < 0.05)")
        for var1, var2, p_val, strength, chi2_stat, dof_val in chi2_results:
            color = get_strength_color(strength)
            st.markdown(
                f"- **{var1}** vs **{var2}** → `p-value = {p_val:.4f}`, "
                f"`chi² = {chi2_stat:.2f}`, `df = {dof_val}` "
                f"(<span style='color:{color}'><strong>{strength}</strong></span>)",
                unsafe_allow_html=True
            )
    else:
        st.info("No significant associations found using Chi-Square test.")

    # Section: ANOVA Test with Dynamic Explanation
    st.header("📏 ANOVA Test for Challenge, Cost, Time")

    # Preprocess: Encode booleans and normalize
    df_encoded = df.copy()

    # Convert TRUE/FALSE in Proficiency columns to 1/0
    proficiency_cols = [c for c in df.columns if "Proficiency" in c]
    df_encoded[proficiency_cols] = df_encoded[proficiency_cols].apply(
        lambda col: col.map(lambda x: 1 if x is True else 0 if x is False else np.nan)
    )

    # Normalize Games, Cost, and Time by Experience_Years
    for prefix in ["Games", "Cost", "Time"]:
        for c in df.columns:
            if c.startswith(prefix):
                df_encoded[f"{c}_PerYear"] = df[c] / df["Experience_Years"].replace(0, np.nan)

    # ANOVA Targets
    anova_targets = [c for c in df_encoded.columns if any(x in c for x in ["Challenge", "Cost", "Time"])]
    grouping_vars  = ["Skill_Level", "Preferred_Platform", "Recommended_for_Beginners"]

    def generate_explanation(target, cat, f_val, p_val):
        topic = ("difficulty" if "Challenge" in target
                else "cost" if "Cost" in target
                else "time")
        if p_val < 0.05:
            return (
                f"The ANOVA test for **{topic}** ({target}) across **{cat}** levels "
                f"found a significant difference *(F={f_val:.2f}, p={p_val:.4f})*. "
                f"This suggests that the {topic} varies by {cat}."
            )
        else:
            return (
                f"The ANOVA test for **{topic}** ({target}) across **{cat}** levels "
                f"did not find a significant difference *(F={f_val:.2f}, p={p_val:.4f})*. "
                f"This implies similar average {topic} across {cat} groups."
            )

    for target in anova_targets:
        for cat in grouping_vars:
            # skip if only one category level
            if df_encoded[cat].nunique(dropna=True) <= 1:
                continue

            # prepare groups
            grouped = df_encoded[[cat, target]].dropna().groupby(cat)
            groups  = [g[target].values for _, g in grouped if len(g) > 1 and np.std(g[target]) > 0]
            if len(groups) <= 1:
                continue

            # run ANOVA
            try:
                f_val, p_val = f_oneway(*groups)
                explanation = generate_explanation(target, cat, f_val, p_val)
                st.write(explanation, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error in ANOVA for {target} by {cat}: {e}")


    # Section: Regression
    st.header("📉 Regression")
    target = st.selectbox("Dependent variable (target)", numerical_cols)
    predictor = st.selectbox("Independent variable (predictor)", [col for col in numerical_cols if col != target])
    if target and predictor:
        X = sm.add_constant(df[predictor])
        y = df[target]
        model = sm.OLS(y, X, missing='drop').fit()
        st.text(model.summary())

    # T-Test for binary categorical variables only (e.g., Proficiency_NC, etc.)
    st.header("⚖️ Στοχευμένο T-Test")

    # Safe conversion of Proficiency columns from string "TRUE"/"FALSE" to boolean
    bool_cols = [col for col in df.columns if col.startswith("Proficiency")]

    for col in bool_cols:
        df[col] = df[col].apply(lambda x: True if str(x).strip().upper() == "TRUE"
                                else False if str(x).strip().upper() == "FALSE"
                                else np.nan)

    # Identify binary (boolean or 2-value) columns
    selected_cats = [col for col in bool_cols if df[col].nunique() == 2]
    filtered_nums = [col for col in numerical_cols if col != "Developer_ID"]

    for cat in selected_cats:
        for num in filtered_nums:
            group_vals = df[cat].dropna().unique()
            group1 = df[df[cat] == group_vals[0]][num].dropna()
            group2 = df[df[cat] == group_vals[1]][num].dropna()

            if len(group1) > 2 and len(group2) > 2:
                try:
                    t_stat, p_val = ttest_ind(group1, group2)
                    mean1 = group1.mean()
                    mean2 = group2.mean()
                    higher = group_vals[0] if mean1 > mean2 else group_vals[1]
                    lower = group_vals[1] if mean1 > mean2 else group_vals[0]
                    diff = abs(mean1 - mean2)

                    if p_val < 0.05:
                        st.success(
                            f"**Significant difference** in `{num}` based on `{cat}` "
                            f"*(t={t_stat:.2f}, p={p_val:.4f})*.\n"
                            f"The `{higher}` group has a higher average by **{diff:.2f}** than the `{lower}` group."
                        )
                    else:
                        st.info(
                            f"No significant difference found in `{num}` between `{cat}` groups "
                            f"*(t={t_stat:.2f}, p={p_val:.4f})*."
                        )
                except Exception as e:
                    st.warning(f"Error in T-Test for {num} by {cat}: {e}")
            else:
                st.warning(f"Not enough data points for `{num}` by `{cat}`.")

# --- README TAB ---
def read_me():
    st.subheader("📖 README: In-Depth Analysis on Video Game Development Techniques")
    st.markdown("""This platform visualizes and analyzes responses, from knowledgeable individuals, on various development techniques, platforms, and tools used in the game industry. Below is a comprehensive guide to understanding how the site works, what it shows, and why it matters.

---

### 🌐 How to Use the Website
Use the sidebar to select filters like type of platform, techniques and the analysis models.
Visualizations will automatically update based on your selections.
Hover over any graph to view exact values and data labels.
Navigate through the tabs to access different types of insights.

---

### 📚 Libraries Used & Their Purpose
**Pandas**: For efficient data wrangling and manipulation.
**NumPy**: To support numerical operations and optimize performance.
**Matplotlib & Seaborn**: For building intuitive, beautiful visualizations.
**Scikit-learn**: Used in correlation calculations and analytical modeling.
**Streamlit**: For building this fast, interactive, browser-based UI.

---

### 🧠 Data Collection & Processing
Data is sourced from questionnaires answered by seasoned game developers.
All responses are cleaned, standardized, and structured for analysis.
Key operations include string normalization, handling nulls, and transforming categorical values.
Aggregation and correlation calculations help expose meaningful patterns.

---

### 📈 Graphs & What They Show
**Correlation Heatmaps**: Reveal relationships (positive/negative) between tools, communication, and development success.
**Bar Charts & Histograms**: Show frequency of usage and preferences across platforms or tools.
**Categorical Visuals**: Compare data grouped by experience level or development environment.
**Interactive Filters**: Allow real-time comparison of multiple variables.

---

### 🔍 Scientific Insights
Emphasis on **correlation over causation** ensures a data-driven yet cautious interpretation.
Insights are **experience-weighted**, as all participants are active or former professionals.
Visuals are tailored to surface **practical conclusions** about what works and what doesn't.

---

### 🧪 Why This Tool Matters
**Developers**: Learn which workflows and environments lead to efficient results.
**Researchers**: Explore a validated dataset to fuel further study.
**Managers/Teams**: Make informed decisions on technology stacks and team structures.

---

### 🚀 Final Words
This app is not just about data — it's about discovering **patterns, relationships**, and **best practices** in game development, through a scientific and visually engaging lens.

Feel free to explore, analyze, and draw your own conclusions from the wealth of experience encapsulated in this tool.
                

---
                
### ❕ Last Note for Potential Improvements
The data from the questionnaire are limited to a small amount, hence why we created the synthetic data sheet. 
Ideally, we would like to gather more types of data at a bigger amount for a better understanding of the real world dynamics between them, such as team size, 
experience in years, developer frequency or anything else that could affect the time, cost and challenge of a game's development and analyze again for more accurate results.
""")

# --- Tab Navigation ---
if st.session_state.get("main_tab") is None:
    active_tab = st.selectbox("Select Tab", tabs, key="main_tab")
else:
    active_tab = st.session_state.main_tab

if active_tab == "Application":
    application()
elif active_tab == "Statistical Analysis":
    statistical_analysis()
elif active_tab == "README":
    read_me()
