# Copyright 2022 Abhishek Gupta, Rishikesh Ingale and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
pip3 install tensorflow-io
