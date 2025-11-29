#!/bin/bash
# Monitor pipeline execution on EC2
# Usage: ./monitor_pipeline.sh EC2_IP

EC2_IP=${1:-"35.153.53.133"}
KEY_PATH="$HOME/.ssh/spotify-analyser-key.pem"
EC2_USER="ubuntu"

echo "========================================="
echo "📊 Pipeline Monitoring Dashboard"
echo "========================================="
echo "EC2: $EC2_IP"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "========================================="
    echo "📊 PIPELINE STATUS - $(date '+%H:%M:%S')"
    echo "========================================="
    echo ""
    
    # Check if pipeline is running
    PIPELINE_RUNNING=$(ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "ps aux | grep 'master_pipeline.py' | grep -v grep" 2>/dev/null)
    
    if [ -z "$PIPELINE_RUNNING" ]; then
        echo "❌ Pipeline not running"
        echo ""
        echo "Checking if it completed..."
        ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "tail -20 ~/spotify_analysis/logs/pipeline_summary.txt 2>/dev/null || echo 'No summary found'"
        break
    else
        echo "✅ Pipeline is RUNNING"
        echo ""
        
        # Show process info
        echo "🔧 Process Info:"
        echo "$PIPELINE_RUNNING" | awk '{printf "   PID: %s | CPU: %s%% | Memory: %s%%\n", $2, $3, $4}'
        echo ""
        
        # Show latest log entries
        echo "📝 Latest Activity (data_cleaning.log):"
        ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "tail -5 ~/spotify_analysis/logs/data_cleaning.log 2>/dev/null" | sed 's/^/   /'
        echo ""
        
        # Show disk usage
        echo "💾 Disk Usage:"
        ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "df -h /home/ubuntu | tail -1" | awk '{printf "   Used: %s / %s (%s)\n", $3, $2, $5}'
        echo ""
        
        # Show memory usage
        echo "🧠 Memory Usage:"
        ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "free -h | grep Mem" | awk '{printf "   Used: %s / %s\n", $3, $2}'
        echo ""
        
        # Check output files
        echo "📁 Output Files:"
        ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "ls -lh ~/spotify_analysis/data/processed/*.{csv,parquet} 2>/dev/null | tail -5" | awk '{printf "   %s - %s\n", $9, $5}' || echo "   No processed files yet"
        echo ""
        
        echo "========================================="
        echo "Next update in 60 seconds..."
        echo "Press Ctrl+C to stop monitoring"
        echo "========================================="
    fi
    
    sleep 60
done

