# bike-cveillance

## RPi setup

- flash Raspian Lite on an SD card
- mount the SD card and `touch ssh` in the boot partition
- continue with below steps to set up the OS and clone the repo


```shell
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config -y
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt-get install libxvidcore-dev libx264-dev -y
sudo apt-get install libgtk2.0-dev libgtk-3-dev -y
sudo apt-get install libatlas-base-dev gfortran -y

sudo apt install git
sudo apt install python3-pip -y
git clone https://github.com/sebas-seck/bike-cveillance.git
pip3 install opencv-python
```

Replace the contents of `~/.profile/ with:

```bash
PATH="$HOME/.local/bin:$PATH"
export PATH
```

```shell
pip3 install pip-tools
cd bike-cveillance
pip-compile --output-file=linux-armhf-py3.7-requirements.txt linux-armhf-py3.7-requirements.in
pip3 install -r linux-armhf-py3.7-requirements.txt
```