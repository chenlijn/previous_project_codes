cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir

cd $root_dir

redo=1
data_root_dir="/root/mount_out/data"
dataset_name="slit"
mapfile="$root_dir/labelmap_pupil.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=500 # input size
height=500

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test train
do
  python /root/caffe/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/$subset.txt $root_dir/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
