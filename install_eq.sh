if [ -d "$HOME/miniconda3" ]; then
    echo "Miniconda3 is available in the home directory."

    echo 'source ~/miniconda3/bin/activate' >> ~/.bashrc
    source ~/.bashrc

    # Exit immediately if a command exits with a non-zero status
    set -e

    # Source Conda initialization script
    source ~/miniconda3/etc/profile.d/conda.sh  # Adjust this path as necessary


    echo "-------------------------installing equidiff------------------------"

    git clone https://github.com/pointW/equidiff.git
    cd equidiff

    conda env create -f conda_environment.yaml --solver=libmamba -y 

    conda activate equidiff


    echo "------------------------- Installing Mimicgen -------------------------"

    # Navigate to the parent directory
    cd ~

    # Clone the Mimicgen repository if it doesn't already exist
    if [ ! -d "mimicgen_environments" ]; then
        git clone https://github.com/NVlabs/mimicgen_environments.git
    else
        echo "Mimicgen repository already exists. Skipping clone."
    fi

 
    cd mimicgen_environments 
    git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9 
    pip install -e . 
    cd ~/equidiff



    echo "-------------------------installing huggingface_hub fix------------------------"
    pip install huggingface_hub==0.25.0

    echo "------------------------- done ------------------------"
else
    echo "Miniconda3 is not found in the home directory."

    sudo apt-get update
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    sudo apt install -y gfortran
    sudo apt-get install libegl1-mesa-dev



    echo "-------------------------installing miniconda------------------------"
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh


    echo 'source ~/miniconda3/bin/activate' >> ~/.bashrc
    source ~/.bashrc

    #then restart the terminal


fi

