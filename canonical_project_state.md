# UDE Points — Canonical Project State

**Baseline date: 21 August 2026**

This document is intended to be the **authoritative compact context** for continuing the UDE Points project. It preserves important decisions, current architecture, validated findings, known uncertainties, and terminology. Historical conversation should not override this document or the current source files.

Be certain to keep your responses succinct and efficient.

---

## 1. Project Objective & Design Scope

**UDE Points** is a custom UFC fighter-ranking system designed to quantify a fighter's historical career strength and resume quality by evaluating not merely whether they won, but **how they won, how dominant they were (PDI), who they beat, and the circumstances surrounding the fight**.

### Primary Goals
* **Who is the UFC's greatest ever fighter (GOAT status)?**
* **Who is the UFC's current best fighter (P4P resume status)?**

### Design Scope & Non-Goals
* **Evaluative, Not Predictive:** The metric evaluates accumulated fight-by-fight historical merit. It is **not** a predictive fight-forecasting engine, nor a standard zero-sum Elo ladder.
* **Non-Redundant Signals:** Different components capture distinct information rather than repeatedly rewarding the same underlying phenomenon.
* **Volume-Bias & Inflation Control:** Downstream ranking layers (`rank_fighters_by_shrunk_ude_rate`) handle volume normalization and shrinkage, keeping the per-fight scoring loop strictly focused on fight performance.

---

## 2. Recent Architecture Upgrades

1. **Symmetric Loss Mechanics:** 
   - **Performance Scaling (`perf_scale`):** Extends to losses ($0.20\text{--}1.0$), ensuring stoppage finishes expose a fighter to more penalty than tight split decisions.
   - **Asymmetric Opponent Quality ($M_{\text{loss}}$):** Losing to elite opposition carries no artificial cushion ($1.0\times$ baseline penalty), while losing to low-tier opposition incurs an amplified loss penalty ($1.0\text{--}1.5\times$).

2. **Continuous Tanh Gap Smoothing:**
   - Replaced discrete rating gap steps ($\ge 30, 40, 50, 60$) with a continuous, bounded $\tanh$ S-curve function to eliminate threshold cliffs and mathematical artifacts.
   - **Trailing-3 Rating Gap:** The upset bonus uses a symmetric trailing-3 fight rolling average for both the fighter's and opponent's pre-fight ratings to prevent path-dependent post-win penalty traps.

---

## 3. Core Data & Architecture

The principal dataset contains **8,564 UFC fights** through the Islam Makhachev vs. Ian Machado Garry timeframe.

### Fundamental Dataframe Structure
The fight dataframe utilizes a standardized two-sided convention:
```text
fighter_1
fighter_2

<feature>_fighter_1
<feature>_fighter_2
```
This naming convention is strictly enforced across utility transformations.

---

## 4. Career Trajectories & Utilities
ude_points_utils.create_fighter_career_dataset converts the two-sided fight dataframe into fighter-specific career trajectories.
```text
fight-level dataset 
        ↓
fighter-level career trajectory 
        ↓
ranking / historical analysis / visualization
```
---

## 5. Current Source Files
* ```text ude_points_algorithm.py ``` — Authoritative UDE scoring implementation.

* ```text ude_points_feature_engineering_pipeline.py ``` — Generates chronological state and PDI fight-performance features.

* ```text ude_points_utils.py ``` — Handles peak, career, shrunk-rate rankings, and career dataset conversions.

* ```text latest_fights_up_to_islam_garry_with_ude_points_calculated_v2_5 ``` — Main historical fight dataset with calculated UDE points.

* ```text all_fights_data_processed_engineered_and_ready_for_ude_points ``` — Pre-scored feature-engineered dataset.
