conda install python=3.10 -y
apt update -y
apt-get install libgl1-mesa-glx -y
pip install -U magic-pdf[full] detectron2 opencv-python-headless --extra-index-url https://wheels.myhloli.com --cache-dir /data-pfs/jd/cache/pip
export PYTHONPATH=$PYTHONPATH:/data-pfs/jd/programs/Table_Rec

# det e create config.yml <workdir> 会强制改变工作目录， 但是上传文件时会上传所有<workdir>下的文件
#cd /data-pfs/jd/programs/Table_Rec

cd /data-pfs/jd/programs/Table_Rec
echo "current folder"
pwd
python /data-pfs/jd/programs/Table_Rec/scripts/extract_tables_mp.py