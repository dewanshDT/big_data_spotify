#!/bin/bash

################################################################################
# EC2 Environment Setup Script
# Run this script on your EC2 instance after SSH connection
################################################################################

echo "=========================================="
echo "Spotify Data Analysis - EC2 Setup"
echo "=========================================="

# Update system
echo ""
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Python 3.11
echo ""
echo "🐍 Installing Python 3.11..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip
echo ""
echo "📦 Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install git
echo ""
echo "📦 Installing git..."
sudo apt-get install -y git

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get install -y build-essential libssl-dev libffi-dev

# Create project directory
echo ""
echo "📁 Creating project directory..."
mkdir -p ~/spotify_analysis
cd ~/spotify_analysis

# Create virtual environment
echo ""
echo "🔧 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install AWS CLI
echo ""
echo "☁️  Installing AWS CLI..."
pip install awscli boto3

# Configure AWS (interactive)
echo ""
echo "🔑 Configuring AWS credentials..."
echo "Please enter your AWS credentials:"
aws configure

# Install Python packages
echo ""
echo "📚 Installing Python packages..."
echo "This may take a few minutes..."

pip install pandas numpy matplotlib seaborn plotly
pip install scikit-learn xgboost
pip install jupyter jupyterlab ipywidgets
pip install pyyaml tqdm python-dotenv

# Install optional packages for big data
echo ""
read -p "Install PySpark for very large data processing? (y/n): " install_spark
if [ "$install_spark" == "y" ]; then
    echo "📦 Installing PySpark..."
    sudo apt-get install -y openjdk-11-jdk
    pip install pyspark pyarrow
fi

# Setup Jupyter
echo ""
echo "📓 Setting up Jupyter..."
jupyter notebook --generate-config

# Create Jupyter config with password
python3 << END
from jupyter_server.auth import passwd
password = input("Enter password for Jupyter (press Enter to skip): ")
if password:
    hashed = passwd(password)
    config_file = "~/.jupyter/jupyter_notebook_config.py"
    with open(config_file.replace("~", "$HOME"), "a") as f:
        f.write(f"\nc.NotebookApp.password = '{hashed}'\n")
        f.write("c.NotebookApp.ip = '0.0.0.0'\n")
        f.write("c.NotebookApp.open_browser = False\n")
        f.write("c.NotebookApp.port = 8888\n")
    print("✅ Jupyter configured with password")
else:
    print("⚠️  Jupyter running without password (not recommended for production)")
END

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,results}
mkdir -p notebooks
mkdir -p scripts
mkdir -p ml_models
mkdir -p visualizations

# Download project files (if git repo available)
echo ""
read -p "Do you have a git repository to clone? (y/n): " has_repo
if [ "$has_repo" == "y" ]; then
    read -p "Enter git repository URL: " repo_url
    git clone $repo_url
fi

# Display info
echo ""
echo "=========================================="
echo "✅ EC2 Setup Complete!"
echo "=========================================="
echo ""
echo "Environment Details:"
echo "  Python version: $(python3 --version)"
echo "  Pip version: $(pip --version)"
echo "  Virtual env: ~/spotify_analysis/venv"
echo ""
echo "Next Steps:"
echo ""
echo "1. Activate virtual environment:"
echo "   source ~/spotify_analysis/venv/bin/activate"
echo ""
echo "2. Download data from S3:"
echo "   python scripts/download_from_s3.py"
echo ""
echo "3. Run data processing:"
echo "   python scripts/03_data_cleaning.py"
echo ""
echo "4. Start Jupyter for analysis:"
echo "   jupyter notebook"
echo "   Then access: http://YOUR_EC2_IP:8888"
echo ""
echo "⚠️  Security Note:"
echo "   Make sure port 8888 is open in your EC2 security group"
echo "   for Jupyter access"
echo ""
echo "=========================================="

# Reminder about security group
echo ""
echo "🔒 Security Group Checklist:"
echo "  [ ] Port 22 (SSH) - for terminal access"
echo "  [ ] Port 8888 (Jupyter) - for notebook access"
echo "  [ ] Restrict IPs to your own for security"
echo ""



