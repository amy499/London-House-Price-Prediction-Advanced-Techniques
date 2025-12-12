# -*- coding: utf-8 -*-
"""Step 10: Final Push - 10 Folds + Maximum Models"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
import random
import gc

warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")

seed = 927
random.seed(seed)
np.random.seed(seed)

print("="*70)
print("Strategy:")
print("  1. 10-Fold CV (better stability)")
print("  2. 15 models (9 CatBoost + 6 LightGBM)")
print("  3. Fine-tuned iterations per model")
print("  4. Optimal weight search")
print("="*70)

# Data
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"\nTrain shape: {data.shape}, Test shape: {test.shape}")

# Handle missing values
dt = []
for i in test.isnull().sum().items():
    if 0 < i[-1] <= len(test) * 0.5:
        if test[i[0]].dtype == object:
            test[i[0]] = test[i[0]].fillna(test[i[0]].mode()[0])
        else:
            test[i[0]] = test[i[0]].fillna(test[i[0]].median())
    if i[-1] > len(test) * 0.5:
        dt.append(i[0])

for i in data.isnull().sum().items():
    if 0 < i[-1] <= len(data) * 0.5:
        if data[i[0]].dtype == object:
            data[i[0]] = data[i[0]].fillna(test[i[0]].mode()[0])
        else:
            data[i[0]] = data[i[0]].fillna(test[i[0]].median())
    if i[-1] > len(data) * 0.5:
        dt.append(i[0])

data = data.drop(columns=dt)
del dt

# Feature engineering
def add_features(df):
    df['total_rooms'] = df['bedrooms'] + df['bathrooms'] + df['livingRooms']
    df['sqm_per_bedroom'] = df['floorAreaSqM'] / (df['bedrooms'] + 0.1)
    df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 0.1)
    df['room_density'] = df['total_rooms'] / (df['floorAreaSqM'] + 1)
    df['sqm_per_room'] = df['floorAreaSqM'] / (df['total_rooms'] + 0.1)
    df['living_to_bedroom_ratio'] = df['livingRooms'] / (df['bedrooms'] + 0.1)
    df['bathroom_per_bedroom'] = df['bathrooms'] / (df['bedrooms'] + 0.1)
    df['has_multiple_bathrooms'] = (df['bathrooms'] >= 2).astype(int)
    df['is_large_property'] = (df['floorAreaSqM'] > 100).astype(int)
    df['is_small_property'] = (df['floorAreaSqM'] < 50).astype(int)
    df['is_studio'] = (df['bedrooms'] <= 1).astype(int)
    df['is_family_size'] = (df['bedrooms'] >= 3).astype(int)
    df['room_balance'] = df['bedrooms'] * df['bathrooms'] * df['livingRooms']
    df['sqm_squared'] = df['floorAreaSqM'] ** 2
    df['sqm_cubed'] = df['floorAreaSqM'] ** 3
    df['bedrooms_squared'] = df['bedrooms'] ** 2
    df['bed_x_sqm'] = df['bedrooms'] * df['floorAreaSqM']
    df['bath_x_sqm'] = df['bathrooms'] * df['floorAreaSqM']
    df['living_x_sqm'] = df['livingRooms'] * df['floorAreaSqM']
    df['total_rooms_x_sqm'] = df['total_rooms'] * df['floorAreaSqM']
    df['lat_x_lon'] = df['latitude'] * df['longitude']
    df['distance_from_center'] = np.sqrt((df['latitude'] - 51.5074)**2 + (df['longitude'] + 0.1278)**2)
    df['location_score'] = df['latitude'] + df['longitude']
    df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12)
    return df

data = add_features(data)
test = add_features(test)

X = data.drop(columns=['ID', 'price'])
y = np.log10(data['price'])

cat_features = ['postcode', 'country', 'outcode', 'tenure', 'propertyType', 'currentEnergyRating']
text_features = ['fullAddress']

# Encode for LightGBM
X_encoded = X.copy()
test_encoded = test[X.columns].copy()

for col in cat_features + text_features:
    le = LabelEncoder()
    combined = pd.concat([X[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    X_encoded[col] = le.transform(X[col].astype(str))
    test_encoded[col] = le.transform(test[col].astype(str))

# ========== 10-FOLD STRATIFIED CV ==========
print("\n" + "="*70)
print("USING 10-FOLD STRATIFIED CV")
print("="*70)

price_bins = pd.qcut(data['price'], q=10, labels=False, duplicates='drop')
n_folds = 10
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

print(f"✓ 10 folds (better stability, less variance)")
print(f"✓ Stratified by price deciles")

all_oof = []
all_test = []
model_names = []

# ========== CATBOOST: 9 DIVERSE MODELS ==========
print("\n" + "="*70)
print("TRAINING 9 CATBOOST MODELS")
print("="*70)

catboost_configs = [
    # Config 1-3: Different depths
    {'iterations': 26000, 'learning_rate': 0.045, 'depth': 9, 'l2_leaf_reg': 0.6, 'bagging_temperature': 0.3, 'border_count': 254, 'seed': 927},
    {'iterations': 24000, 'learning_rate': 0.05, 'depth': 8, 'l2_leaf_reg': 0.5, 'bagging_temperature': 0.4, 'border_count': 254, 'seed': 42},
    {'iterations': 22000, 'learning_rate': 0.055, 'depth': 7, 'l2_leaf_reg': 0.4, 'bagging_temperature': 0.5, 'border_count': 128, 'seed': 2024},
    
    # Config 4-6: Different regularization
    {'iterations': 26000, 'learning_rate': 0.043, 'depth': 9, 'l2_leaf_reg': 0.8, 'bagging_temperature': 0.2, 'border_count': 254, 'seed': 123},
    {'iterations': 24000, 'learning_rate': 0.048, 'depth': 8, 'l2_leaf_reg': 0.7, 'bagging_temperature': 0.35, 'border_count': 254, 'seed': 777},
    {'iterations': 25000, 'learning_rate': 0.046, 'depth': 9, 'l2_leaf_reg': 0.5, 'bagging_temperature': 0.4, 'border_count': 254, 'seed': 999},
    
    # Config 7-9: Different bagging
    {'iterations': 24000, 'learning_rate': 0.047, 'depth': 8, 'l2_leaf_reg': 0.6, 'bagging_temperature': 0.6, 'border_count': 128, 'seed': 555},
    {'iterations': 25000, 'learning_rate': 0.044, 'depth': 9, 'l2_leaf_reg': 0.55, 'bagging_temperature': 0.25, 'border_count': 254, 'seed': 333},
    {'iterations': 23000, 'learning_rate': 0.052, 'depth': 8, 'l2_leaf_reg': 0.65, 'bagging_temperature': 0.45, 'border_count': 128, 'seed': 666},
]

for config_idx, config in enumerate(catboost_configs, 1):
    model_seed = config.pop('seed')
    print(f"\n  CatBoost {config_idx}/9 (depth={config['depth']}, lr={config['learning_rate']}, seed={model_seed})")
    
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(test))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, price_bins), 1):
        if fold_idx % 2 == 0:  # Print every 2 folds to reduce output
            print(f"    Folds {fold_idx-1}-{fold_idx}/10... ", end='', flush=True)
        
        model = CatBoostRegressor(
            **config,
            random_state=model_seed + fold_idx,
            task_type='GPU',
            verbose=0
        )
        
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
            cat_features=cat_features,
            text_features=text_features,
            early_stopping_rounds=500,
            verbose=False
        )
        
        oof_pred[valid_idx] = model.predict(X.iloc[valid_idx])
        test_pred += model.predict(test[X.columns]) / n_folds
        
        if fold_idx % 2 == 0:
            mae = mean_absolute_error(y.iloc[valid_idx], oof_pred[valid_idx])
            print(f"MAE: {mae:.6f}")
    
    oof_mae = mean_absolute_error(y, oof_pred)
    print(f"    Overall OOF MAE: {oof_mae:.6f}")
    
    all_oof.append(oof_pred)
    all_test.append(test_pred)
    model_names.append(f"CB{config_idx}")
    gc.collect()

# ========== LIGHTGBM: 6 DIVERSE MODELS ==========
print("\n" + "="*70)
print("TRAINING 6 LIGHTGBM MODELS")
print("="*70)

lgbm_configs = [
    # Config 1-3: Different depths/leaves
    {'n_estimators': 23000, 'learning_rate': 0.038, 'max_depth': 9, 'num_leaves': 80, 'reg_lambda': 0.7, 'reg_alpha': 0.05, 'subsample': 0.75, 'colsample_bytree': 0.75, 'seed': 927},
    {'n_estimators': 21000, 'learning_rate': 0.042, 'max_depth': 8, 'num_leaves': 64, 'reg_lambda': 0.5, 'reg_alpha': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'seed': 2024},
    {'n_estimators': 22000, 'learning_rate': 0.040, 'max_depth': 10, 'num_leaves': 100, 'reg_lambda': 0.8, 'reg_alpha': 0.03, 'subsample': 0.72, 'colsample_bytree': 0.72, 'seed': 555},
    
    # Config 4-6: Different regularization
    {'n_estimators': 23000, 'learning_rate': 0.037, 'max_depth': 9, 'num_leaves': 85, 'reg_lambda': 0.9, 'reg_alpha': 0.02, 'subsample': 0.78, 'colsample_bytree': 0.78, 'seed': 123},
    {'n_estimators': 21000, 'learning_rate': 0.043, 'max_depth': 8, 'num_leaves': 70, 'reg_lambda': 0.6, 'reg_alpha': 0.08, 'subsample': 0.82, 'colsample_bytree': 0.82, 'seed': 777},
    {'n_estimators': 22000, 'learning_rate': 0.039, 'max_depth': 9, 'num_leaves': 75, 'reg_lambda': 0.75, 'reg_alpha': 0.06, 'subsample': 0.76, 'colsample_bytree': 0.76, 'seed': 333},
]

for config_idx, config in enumerate(lgbm_configs, 1):
    model_seed = config.pop('seed')
    print(f"\n  LightGBM {config_idx}/6 (depth={config['max_depth']}, lr={config['learning_rate']}, seed={model_seed})")
    
    oof_pred = np.zeros(len(X))
    test_pred = np.zeros(len(test))
    
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_encoded, price_bins), 1):
        if fold_idx % 2 == 0:
            print(f"    Folds {fold_idx-1}-{fold_idx}/10... ", end='', flush=True)
        
        model = LGBMRegressor(
            **config,
            random_state=model_seed + fold_idx,
            device='gpu',
            verbose=-1
        )
        
        model.fit(
            X_encoded.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X_encoded.iloc[valid_idx], y.iloc[valid_idx])],
            callbacks=[
                __import__('lightgbm').early_stopping(500, verbose=False),
                __import__('lightgbm').log_evaluation(0)
            ]
        )
        
        oof_pred[valid_idx] = model.predict(X_encoded.iloc[valid_idx])
        test_pred += model.predict(test_encoded) / n_folds
        
        if fold_idx % 2 == 0:
            mae = mean_absolute_error(y.iloc[valid_idx], oof_pred[valid_idx])
            print(f"MAE: {mae:.6f}")
    
    oof_mae = mean_absolute_error(y, oof_pred)
    print(f"    Overall OOF MAE: {oof_mae:.6f}")
    
    all_oof.append(oof_pred)
    all_test.append(test_pred)
    model_names.append(f"LGB{config_idx}")
    gc.collect()

# ========== META-MODEL OPTIMIZATION ==========
print("\n" + "="*70)
print("META-MODEL: OPTIMAL WEIGHT SEARCH")
print("="*70)

meta_train = np.column_stack(all_oof)
meta_test = np.column_stack(all_test)

print(f"\nTotal models: {len(all_oof)}")
print("\nIndividual OOF Performance:")
for name, oof in zip(model_names, all_oof):
    mae = mean_absolute_error(y, oof)
    print(f"  {name:8s}: {mae:.6f}")

# Simple average
simple_avg = meta_train.mean(axis=1)
simple_mae = mean_absolute_error(y, simple_avg)
print(f"\nSimple Average: {simple_mae:.6f}")

# Inverse MAE weighting
individual_maes = [mean_absolute_error(y, oof) for oof in all_oof]
inv_weights = np.array([1.0 / mae for mae in individual_maes])
inv_weights = inv_weights / inv_weights.sum()

inv_weighted = (meta_train * inv_weights).sum(axis=1)
inv_mae = mean_absolute_error(y, inv_weighted)
print(f"Inverse MAE:    {inv_mae:.6f}")

# Ridge with different alphas
best_ridge_mae = float('inf')
best_ridge_alpha = None
best_ridge_model = None

for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(meta_train, y)
    ridge_pred = ridge.predict(meta_train)
    ridge_mae = mean_absolute_error(y, ridge_pred)
    
    if ridge_mae < best_ridge_mae:
        best_ridge_mae = ridge_mae
        best_ridge_alpha = alpha
        best_ridge_model = ridge

print(f"Best Ridge (α={best_ridge_alpha}): {best_ridge_mae:.6f}")

# Select best approach
approaches = {
    'Simple Average': (simple_mae, meta_test.mean(axis=1)),
    'Inverse MAE': (inv_mae, (meta_test * inv_weights).sum(axis=1)),
    'Ridge': (best_ridge_mae, best_ridge_model.predict(meta_test))
}

best_approach = min(approaches.items(), key=lambda x: x[1][0])
print(f"\n✓ Best: {best_approach[0]} with MAE: {best_approach[1][0]:.6f}")

# ========== FINAL PREDICTIONS ==========
final_pred_log = best_approach[1][1]
final_prices = 10 ** final_pred_log

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

print(f"\nTest predictions:")
print(f"  Min:    £{final_prices.min():,.0f}")
print(f"  Max:    £{final_prices.max():,.0f}")
print(f"  Mean:   £{final_prices.mean():,.0f}")
print(f"  Median: £{np.median(final_prices):,.0f}")

submission = pd.DataFrame({
    'ID': test['ID'],
    'price': final_prices
})

submission.to_csv('submission_step10_final_push.csv', index=False)

print("\n" + "="*70)
print("✓ STEP 10 COMPLETE: FINAL PUSH")
print("="*70)
print(f"\nPrevious: 152,895 MAE")
print(f"This OOF: {best_approach[1][0]:.6f}")
