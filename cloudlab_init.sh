sudo apt install python3-venv
sudo apt install python3-dev

mkdir project
cd project
python3-m venv env_cerebro
source env_cerebro/bin/activate

pip3 install --upgrade pip
pip3 install tensorflow==2.3.0
pip3 install cython

git clone https://github.com/Abhishek2304/cerebro-system.git && cd cerebro-system && make

pip3 install -U 'ray[default]'
pip3 install ray[data]
