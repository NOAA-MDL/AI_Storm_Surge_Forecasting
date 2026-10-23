#!/bin/bash
#SBATCH --job-name=Storm_Classifier_2D_CNN_20var
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/contrib/Mamoudou.Ba/logs/classifier_all_filters_pipeline_%j.log
#SBATCH --error=/contrib/Mamoudou.Ba/logs/classifier_all_filters_pipeline_%j.err

BASE_DIR="/contrib/Mamoudou.Ba"
CONDA_LIB="$BASE_DIR/miniconda3/envs/mlaws2t/lib/python3.11/site-packages/nvidia"

# 🌟 DYNAMICALLY LINK ALL NVIDIA CUDA 11 SHARED OBJECTS (.so FILES)
export LD_LIBRARY_PATH="$CONDA_LIB/cuda_runtime/lib:$CONDA_LIB/cudnn/lib:$CONDA_LIB/cublas/lib:$CONDA_LIB/cufft/lib:$CONDA_LIB/curand/lib:$CONDA_LIB/cusolver/lib:$CONDA_LIB/cusparse/lib:$LD_LIBRARY_PATH"

export TF_ENABLE_ONEDNN_OPTS=0
unset CUDA_VISIBLE_DEVICES

# Activate Conda Environment
source /contrib/Mamoudou.Ba/miniconda3/etc/profile.d/conda.sh
conda activate mlaws2t

BASE_DIR="/contrib/Mamoudou.Ba/storm_surge_scripts/"
LOG_DIR="$BASE_DIR/logs"

# 🌟 Active loop-optimized python script definitions
#PIVOT_SCRIPT="gru_obs_offshore_feature_gfs_etss_storms_weather_pivot_multi_stations_multi_cycles_training.py"
PIVOT_SCRIPT="gru_gfs_weather_obs_etss_water_level_gfs_etss_water_level_multi_cycle_rolling_5gfs_training.py"


# 🌟 Load Cluster CUDA Modules (fallback to path if module fails)
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null

# Export environment hooks for CUDA drivers
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Prevent thread-stacking memory locks inside TensorFlow
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "🚀 Launching Automated Multi-Station observed storms Training Suite..."
echo "📊 Iterating lookbacks [24, 48] and cycles [00Z, 06Z, 12Z, 18Z] across all configured stations..."

# =====================================================================
# 🟦 STEP 1: Execute the PIVOT Framework Models (All Stations Loop)
# =====================================================================
echo "🔄 [1/4] Running PIVOT Multi-Station Architecture Target..."
$CONDA_PYTHON "$BASE_DIR/$PIVOT_SCRIPT" > "$LOG_DIR/all_stations_pivot_observed_storms_training.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Pipeline failed at global PIVOT observed storms. Check error logs."
    exit 1
fi

# =====================================================================
# 🟪 STEP 2: Execute the DIRECT Framework Models (Commented Out)
# =====================================================================
# echo "🔄 [3/4] Running DIRECT Multi-Station Architecture Target..."
# $CONDA_PYTHON "$BASE_DIR/$DIRECT_SCRIPT" > "$LOG_DIR/all_stations_direct_resid_training.log" 2>&1
# if [ $? -ne 0 ]; then
#     echo "❌ Pipeline failed at global DIRECT RESID execution phase. Check error logs."
#     exit 1
# fi

# echo "🔄 [4/4] Running DIRECT Multi-Station Architecture with BIAS Target..."
# $CONDA_PYTHON "$BASE_DIR/$DIRECT_SCRIPT" > "$LOG_DIR/all_stations_direct_bias_training.log" 2>&1
# if [ $? -ne 0 ]; then
#     echo "❌ Pipeline failed at global DIRECT BIAS execution phase. Check error logs."
#     exit 1
# fi

# =====================================================================
# 🎉 SUCCESS TERMINATION
# =====================================================================
echo "✅ Deep learning structural matrices compiled successfully for all active stations!"
echo "💾 H5 weights, JSON models, and RobustScalers written cleanly to ../storm_surge/"
exit 0
