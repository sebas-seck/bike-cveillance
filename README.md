# bike-cveillance

## RPi setup

> tested on a Raspberry 3 B+

- flash Raspian Lite on an SD card
- mount the SD card and `touch ssh` in the boot partition
- continue with below steps to set up the OS and clone the repo


```shell
sudo apt-get update
sudo apt-get upgrade
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
rfkill unblock 0
```

```bash
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=DE

network={
    ssid="YOURSSID"
    psk="YOURPASSWORD"
    scan_ssid=1
}
```


```shell
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

### yolo V3
wget -O /home/pi/bike-cveillance/models/yolo/yolov3-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
wget -O /home/pi/bike-cveillance/models/yolo/yolov3.weights https://pjreddie.com/media/files/yolov3.weights

### mobilenet SSD
mkdir models/ssd_mobilenet

wget -O models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt

wget -O models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

tar -xf models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.tar.gz -C models/ssd_mobilenet/
