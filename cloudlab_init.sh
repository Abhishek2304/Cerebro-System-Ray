sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install python3-venv
sudo apt-get -y install python3-dev

mkdir project
cd project
python3 -m venv env_cerebro
source env_cerebro/bin/activate

pip3 install --upgrade pip
pip3 install tensorflow==2.3.0
pip3 install cython
pip3 install py4j==0.10.9.2

git clone https://github.com/Abhishek2304/cerebro-system.git && cd cerebro-system && make

pip3 install -U 'ray[default]'
pip3 install ray[data]
pip3 install pandas==1.1.0
pip3 install pickle5==0.0.10
