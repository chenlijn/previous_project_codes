
src_dir=./data2
dst_dir=./data2_txt
categ1=healthy
categ2=ill
categ3=after_surgery

img_t1=mydriasis_diffuse_light
img_t2=mydriasis_slit_light
img_t3=small_pupil_diffuse_light
img_t4=small_pupil_slit_light

categs=(`ls $src_dir`)
for categ in ${categs[*]}
do
	img_ts=(`ls $src_dir/$categ`)
	for img_t in ${img_ts[*]}
	do
		#filename=""
		#filename+=$catg
		find $PWD/$src_dir/$categ/$img_t -type f -name '*jpg' > $dst_dir/$categ"_"$img_t.txt
	done
done



#cd src_dir/categ1/

#cd -
