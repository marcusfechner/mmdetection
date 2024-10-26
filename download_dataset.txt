conda install axel -c conda-forge
cd /pool/bs4370/datasets/data
mkdir imagenet
cd imagenet
axel -n 10 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
axel -n 10 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

wget https://raw.githubusercontent.com/pytorch/examples/refs/heads/main/imagenet/extract_ILSVRC.sh
chmod +x extract_ILSVRC.sh
./extract_ILSVRC.sh
