

mkdir data/cityscapes/
cd data/cityscapes/
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=aanani@mpi-sws.org&password=mypassword=Login' https://www.cityscapes-dataset.com/login/
# refer to https://github.com/cemsaz/city-scapes-script
# get gtFine data/gtFine_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
unzip gtFine_trainvaltest.zip
rm -rf gtFine_trainvaltest.zip
# get  leftImg8bit data/leftImg8bit_trainvaltest.zip
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip leftImg8bit_trainvaltest.zip
rm -rf leftImg8bit_trainvaltest.zip

cd ../..
mkdir data/acdc/
cd data/acdc