dataroot="data"
snetroot="${dataroot}/ShapeNet/ShapeNetCore.v1"

# for cat in "03001627" # just the chair
for cat in "04530566" "04090263" "03211117" "03636649" "03691459" "02933112" "03001627" "02828884" "02958343" "02691156" "04256520" "04379243" "04401088"
do
    cmd="unzip -q -d ${snetroot} ${snetroot}/${cat}.zip"
    echo $cmd
    $cmd
    echo "${cat} done!"
done
echo "All done!"