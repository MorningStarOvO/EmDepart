#!/bin/bash 

cd data 

# ----- download CUB ----- # 
# ===============================================================
# https://pan.baidu.com/s/1o60hA0qrupDjtMGPVCke3A
# u0sr
# ===============================================================
# ===============================================================
# https://data.caltech.edu/records/65de6-vp158
# ===============================================================

# tar zxvf /root/autodl-tmp/CUB2002011/CUB_200_2011.tgz
# tar zxvf /root/autodl-tmp/CUB2002011/segmentations.tgz

# ----- AWA2 ----- # 
echo "download AWA2 dataset !"
mkdir AWA2
cd AWA2

wget https://cvml.ista.ac.at/AwA2/AwA2-data.zip 
unzip AwA2-data.zip
rm -rf AwA2-data.zip

cd .. 

# ----- FLO ----- # 
echo "download FLO dataset !"
mkdir FLO 
cd FLO 

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz 
tar zxvf 102flowers.tgz 
rm -rf 102flowers.tgz 

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat 
wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat 

cd .. 

