#!/bin/bash
# Run hyperparameter configuration comparison
# Tests 5 configurations and stores results

set -e

VENV="/home/sergio/ai-trader/venv/bin/python"
SCRIPT_DIR="/home/sergio/ai-trader/backend/scripts"
CONFIG_DIR="/home/sergio/ai-trader/backend/configs"
RESULTS_FILE="/home/sergio/ai-trader/backend/data/hpo_comparison_results.txt"

echo "======================================================================" | tee $RESULTS_FILE
echo "HYPERPARAMETER CONFIGURATION COMPARISON" | tee -a $RESULTS_FILE
echo "Started: $(date)" | tee -a $RESULTS_FILE
echo "======================================================================" | tee -a $RESULTS_FILE

# Function to create config and run test
run_config() {
    local name=$1
    local desc=$2
    local h1_est=$3
    local h1_depth=$4
    local h1_lr=$5
    local h4_est=$6
    local h4_depth=$7
    local h4_lr=$8
    local d_est=$9
    local d_depth=${10}
    local d_lr=${11}

    echo ""
    echo "======================================================================" | tee -a $RESULTS_FILE
    echo "Testing: $name" | tee -a $RESULTS_FILE
    echo "Description: $desc" | tee -a $RESULTS_FILE
    echo "1H: n_est=$h1_est, depth=$h1_depth, lr=$h1_lr" | tee -a $RESULTS_FILE
    echo "4H: n_est=$h4_est, depth=$h4_depth, lr=$h4_lr" | tee -a $RESULTS_FILE
    echo "D:  n_est=$d_est, depth=$d_depth, lr=$d_lr" | tee -a $RESULTS_FILE
    echo "======================================================================" | tee -a $RESULTS_FILE

    # Create config file
    cat > $CONFIG_DIR/optimized_hyperparams.json << EOF
{
    "optimization_date": "$(date -Iseconds)",
    "test_config": "$name",
    "results": {
        "1H": {
            "best_params": {
                "n_estimators": $h1_est,
                "max_depth": $h1_depth,
                "learning_rate": $h1_lr,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "gamma": 0.1
            }
        },
        "4H": {
            "best_params": {
                "n_estimators": $h4_est,
                "max_depth": $h4_depth,
                "learning_rate": $h4_lr,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "gamma": 0.1
            }
        },
        "D": {
            "best_params": {
                "n_estimators": $d_est,
                "max_depth": $d_depth,
                "learning_rate": $d_lr,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "gamma": 0.1
            }
        }
    }
}
EOF

    # Train
    echo "Training $name..."
    $VENV $SCRIPT_DIR/train_mtf_ensemble.py \
        --sentiment --stacking --use-rfecv --use-optimized-params \
        --output models/hpo_test_$name 2>&1 | tail -30

    # Backtest
    echo "Backtesting $name..."
    $VENV $SCRIPT_DIR/backtest_mtf_ensemble.py \
        --model-dir models/hpo_test_$name 2>&1 | tee -a $RESULTS_FILE | grep -E "(Total Pips|Win Rate|Profit Factor|Total Trades)"
}

# Run baseline (current production params)
run_config "baseline" "Current production configuration" \
    200 6 0.05 \
    150 5 0.05 \
    100 4 0.05

# Run conservative (more regularized)
run_config "conservative" "More regularized (less depth, fewer trees, lower LR)" \
    150 5 0.03 \
    120 4 0.03 \
    80 3 0.03

# Run deeper (slightly more capacity)
run_config "deeper" "Slightly more capacity (depth +1)" \
    200 7 0.05 \
    150 6 0.05 \
    100 5 0.05

# Run more_trees (more trees, lower LR)
run_config "more_trees" "More trees with lower learning rate" \
    300 6 0.03 \
    225 5 0.03 \
    150 4 0.03

# Run shallow_fast (shallow but more trees, higher LR)
run_config "shallow_fast" "Shallow trees with higher learning rate" \
    250 4 0.08 \
    200 3 0.08 \
    150 3 0.08

echo ""
echo "======================================================================" | tee -a $RESULTS_FILE
echo "COMPARISON COMPLETE" | tee -a $RESULTS_FILE
echo "Finished: $(date)" | tee -a $RESULTS_FILE
echo "Results saved to: $RESULTS_FILE" | tee -a $RESULTS_FILE
echo "======================================================================" | tee -a $RESULTS_FILE
