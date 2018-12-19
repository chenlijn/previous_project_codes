dst=trained_models_dec_2017
mkdir $dst

wget ftp://172.16.0.14/Projects/Cataract/illness3_mean_diffuse.npy -P $dst
wget ftp://172.16.0.14/Projects/Cataract/illness3_mean_slit.npy -P $dst
wget ftp://172.16.0.14/Projects/Cataract/imaging_types_mean.npy -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet50_imaging_types_iter_175000.caffemodel -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet50_imaging_types_iter_175000.solverstate -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_illness3_diffuse_iter_85000.caffemodel -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_illness3_diffuse_iter_85000.solverstate -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_illness3_slit_iter_85000.caffemodel -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_illness3_slit_iter_85000.solverstate -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_severity_diffuse_iter_40000.caffemodel -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_severity_diffuse_iter_40000.solverstate -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_severity_slit_iter_40000.caffemodel -P $dst
wget ftp://172.16.0.14/Projects/Cataract/resnet_severity_slit_iter_40000.solverstate -P $dst
wget ftp://172.16.0.14/Projects/Cataract/severity_diffuse_mean.npy -P $dst
wget ftp://172.16.0.14/Projects/Cataract/severity_slit_mean.npy -P $dst
