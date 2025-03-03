echo "Start time: $(date)"
START_TIME=$(date +%s)  

if ! command -v conda &> /dev/null; then
    echo "Conda is not initialized, running 'conda init'"
    conda init
    source ~/.bashrc  
fi

echo "GPU Information:"
gpustat  

echo "-------------------------------------------"

cd /path/to/your/directory
mkdir -p logs

echo "-------------------------------------------"
HF_ENDPOINT=xxxxxxxxxxxx CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python src/models.py | tee logs/output_$(date +%Y-%m-%d_%H-%M-%S).txt

echo "-------------------------------------------"
HF_ENDPOINT=xxxxxxxxxxxx CUDA_VISIBLE_DEVICES=0 python src/inference.py | tee logs/output_$(date +%Y-%m-%d_%H-%M-%S).txt

echo "-------------------------------------------"
END_TIME=$(date +%s)  
RUNNING_TIME=$((END_TIME - START_TIME))  

echo "End time: $(date)"
echo "Running time: ${RUNNING_TIME} seconds"
