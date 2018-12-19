
src_dir=./data2_txt
txt_files=(`ls $src_dir`)
recd_file=img_count.txt

for txt_f in ${txt_files[*]}
do
	echo $txt_f >> img_count.txt
	cat $src_dir/$txt_f | wc -l >> img_count.txt
done



