#!/bin/bash

# Script to train all four models sequentially using organized config structure
# Usage: bash train_all_models.sh [additional_args...]
# Example: bash train_all_models.sh --trainer.max_epochs=20

echo "🚀 Starting sequential training of all models..."
echo "Additional args: $@"
echo ""

# Function to run training with error handling
train_model() {
    local model_name="$1"
    local config_path="$2"
    local wandb_path="$3"
    shift 3  # Remove the first 3 arguments, leaving only additional args
    
    echo "=================================================="
    echo "🔄 Training: $model_name"
    echo "=================================================="
    
    # Run the training command with all additional arguments
    python trainer.py fit \
        --config configs/models/config_base.yaml \
        --config "$config_path" \
        --config "$wandb_path" \
        "$@"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_name training completed successfully!"
    else
        echo "❌ $model_name training failed with exit code $exit_code"
        return $exit_code
    fi
    
    echo ""
    return 0
}

# Train all models
start_time=$(date +%s)

echo "📋 Training Queue:"
echo "  1. Baseline Model"
echo "  2. Basic GNN"  
echo "  3. Equivariant GNN"
echo "  4. Large GNN"
echo ""

# Train each model
train_model "Baseline Model" "configs/models/baseline/config.yaml" "configs/models/baseline/wandb.yaml" "$@" || exit 1

train_model "Basic GNN" "configs/models/basic_gnn/config.yaml" "configs/models/basic_gnn/wandb.yaml" "$@" || exit 1

train_model "Equivariant GNN" "configs/models/eq_gnn/config.yaml" "configs/models/eq_gnn/wandb.yaml" "$@" || exit 1

train_model "Large GNN" "configs/models/large_gnn/config.yaml" "configs/models/large_gnn/wandb.yaml" "$@" || exit 1

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

echo "🎉 All models trained successfully!"
echo "⏱️  Total time: ${minutes}m ${seconds}s"
echo ""
echo "📊 Check your W&B dashboard for results:"
echo "   https://wandb.ai/your-username/gnn-center-of-mass" 