
folder=data_test
rm -r $folder
mkdir $folder
cd $folder
rm -r *
mkdir healthy ill after_surgery
cd -

cd $folder/healthy
mkdir mydriasis_diffuse_light mydriasis_slit_light
mkdir small_pupil_diffuse_light small_pupil_slit_light
cd -

cd $folder/ill
mkdir mydriasis_diffuse_light mydriasis_slit_light
mkdir small_pupil_diffuse_light small_pupil_slit_light
cd -

cd $folder/after_surgery
mkdir mydriasis_diffuse_light mydriasis_slit_light
mkdir small_pupil_diffuse_light small_pupil_slit_light
cd -


