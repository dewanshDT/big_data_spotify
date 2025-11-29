#!/bin/bash
# Upload all scripts to EC2
# Usage: ./upload_to_ec2.sh YOUR_EC2_IP

set -e  # Exit on error

if [ -z "$1" ]; then
    echo "❌ Error: EC2 IP address required"
    echo "Usage: ./upload_to_ec2.sh YOUR_EC2_IP"
    echo "Example: ./upload_to_ec2.sh 35.153.53.133"
    exit 1
fi

EC2_IP=$1
KEY_PATH="$HOME/.ssh/spotify-analyser-key.pem"
EC2_USER="ubuntu"
EC2_DIR="~/spotify_analysis"

echo "========================================="
echo "📤 Uploading Scripts to EC2"
echo "========================================="
echo "EC2 IP: $EC2_IP"
echo "Key: $KEY_PATH"
echo ""

# Check if key exists
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ Error: SSH key not found at $KEY_PATH"
    exit 1
fi

echo "✅ SSH key found"
echo ""

# Test connection
echo "🔍 Testing SSH connection..."
if ! ssh -i "$KEY_PATH" -o ConnectTimeout=10 "$EC2_USER@$EC2_IP" "echo 'Connection successful'" 2>/dev/null; then
    echo "❌ Error: Cannot connect to EC2"
    echo "Please check:"
    echo "  1. EC2 instance is running"
    echo "  2. IP address is correct"
    echo "  3. Security group allows SSH from your IP"
    exit 1
fi

echo "✅ SSH connection successful"
echo ""

# Create directories on EC2
echo "📁 Creating directories on EC2..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "mkdir -p ~/spotify_analysis/{data/{raw,processed},scripts,ml_models,visualizations,utils,logs}"
echo "✅ Directories created"
echo ""

# Upload files
echo "📤 Uploading files..."

echo "  → Core scripts..."
scp -i "$KEY_PATH" \
    master_pipeline.py \
    validate_setup.py \
    generate_data_local.py \
    "$EC2_USER@$EC2_IP:$EC2_DIR/"

echo "  → Configuration files..."
scp -i "$KEY_PATH" \
    config.yaml \
    requirements.txt \
    "$EC2_USER@$EC2_IP:$EC2_DIR/"

echo "  → Scripts directory..."
scp -i "$KEY_PATH" scripts/*.py "$EC2_USER@$EC2_IP:$EC2_DIR/scripts/"

echo "  → ML models directory..."
scp -i "$KEY_PATH" ml_models/*.py "$EC2_USER@$EC2_IP:$EC2_DIR/ml_models/"

echo "  → Visualizations directory..."
scp -i "$KEY_PATH" visualizations/*.py "$EC2_USER@$EC2_IP:$EC2_DIR/visualizations/"

echo "  → Utils directory..."
scp -i "$KEY_PATH" utils/*.py "$EC2_USER@$EC2_IP:$EC2_DIR/utils/"

echo ""
echo "========================================="
echo "✅ ALL FILES UPLOADED SUCCESSFULLY!"
echo "========================================="
echo ""
echo "📋 Next Steps:"
echo "1. SSH into EC2:"
echo "   ssh -i $KEY_PATH $EC2_USER@$EC2_IP"
echo ""
echo "2. Navigate to project:"
echo "   cd ~/spotify_analysis"
echo ""
echo "3. Validate setup:"
echo "   source venv/bin/activate"
echo "   python validate_setup.py"
echo ""
echo "4. Generate data on EC2:"
echo "   python3 generate_data_local.py --rows 20000000 --output data/raw/spotify_data.csv --chunk-size 2000000"
echo ""
echo "5. Run pipeline:"
echo "   python master_pipeline.py --skip-download --sample 20000000"
echo ""

