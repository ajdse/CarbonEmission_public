# Carbon Emission Cross-Sectional Feature & Modeling Toolkit
This repository contains a modular, production-grade Python library designed to implement reproducible ML pipelines for predictive analytics, specifically for carbon emission prediction. The architecture prioritizes creating rich, stable features and ensuring MLOps best practices for model validation and deployment.

## Architecture
The primary purpose of this architecture is to ensure unbiased modeling and robust feature engineering at scale on clean, structured data. This package provides a re-usable, configurable framework for transforming input data into production-ready predictive models.

## Technical components/highlights

The core of the library is built using the Scikit-learn API to facilitate pipeline configuration and MLOps compatibility.

1. **Configurable Data Preprocessing Pipeline**
The build_datapreprocessor_pipeline(**kwargs) function dynamically generates a complete feature preprocessing pipeline:

Uses sklearn.ColumnTransformer and sklearn.Pipeline to encapsulate all feature transformations.

Enables configuration-based deployment, allowing feature sets (StandardScaler, OneHotEncoder, TfidfVectorizer) to be modified via dictionary inputs (kwargs), which is ideal for A/B testing and production management.

2. **Advanced Custom Feature Engineering**
The library includes custom Scikit-learn transformers (BaseEstimator, TransformerMixin) for generating stable, predictive features:

IndustryStatsTransformer: Calculates and generates smoothed target-encoded features (Mean/Std Dev) based on a grouping column (e.g., industry). This demonstrates the ability to manage data leakage and utilize statistical regularization to create stable features from high-cardinality categorical data.

WinsorizeTransformer: Provides robust outlier capping functionality, ensuring the model is protected from extreme values in a cross-sectional setting.

3. **Scalable Text Feature Processing**
The pipeline includes modular functions (clean_text) for text normalization, paired with the configurable use of TfidfVectorizer to generate high-dimensional text embeddings, which is critical for incorporating unstructured data sources into the cross-sectional prediction model.

## Usage
Usage is demonstrated by a sample notebook contained in the notebook directory.
