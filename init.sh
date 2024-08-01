# !/bin/bash

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install build-essential libopencv-dev openssh-server

cd ~
if [ ! -f "Miniforge3-$(uname)-$(uname -m).sh" ]; then
    wget https://gh-proxy.com/https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
fi
if [ $? -ne 0 ]; then
    exit 1
fi
chmod +x ./Miniforge3-$(uname)-$(uname -m).sh
rm -rf $HOME/miniforge3
bash ./Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge3

echo '
# >>> conda initialize >>>
#!! Contents within this block are managed by 'conda init'!!
__conda_setup="$("/home/$USER/miniforge3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/$USER/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/$USER/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/$USER/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<' >> ~/.bashrc

__conda_setup="$("/home/$USER/miniforge3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/$USER/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/$USER/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/$USER/miniforge3/bin:$PATH"
    fi
fi

/home/$USER/miniforge3/bin/conda create -n ds python=3.11 -y
if [ $? -ne 0 ]; then
    exit 1
fi

/home/$USER/miniforge3/envs/ds/bin/pip install autopep8 opencv-python pyserial
cd ~
git clone git@github.com:mini-berry/ds.git
