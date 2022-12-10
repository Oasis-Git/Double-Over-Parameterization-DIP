export
echo 'super resolution baseline'
nohup python ../trainer/super_resolution.py --config ../config/super_resolution_baseline.json 2>&1 &

sleep 10s
