python3 src/serve/benchmark.py model_name='vit_small_patch16_224'

python3 src/serve/benchmark.py model_name='mobilenetv3_large_100' Done


model_<batch>_<worker>_feature

 python3 src/serve/batch_server.py batch_deployment=bird_200_mobilenetv3_large_optimized
 python3 src/serve/batch_server.py batch_deployment=food_101_vit_small

 python3 src/serve/batch_server_multiple_worker.py batch_deployment=mobilenetv3_large_100
 python3 src/serve/batch_server_multiple_worker.py batch_deployment=bird_200_mobilenetv3_large_optimized
 python3 src/serve/batch_server_multiple_worker.py batch_deployment=food_101_vit_small ++batch_deployment.workers_per_device=4
 python3 src/serve/batch_server_multiple_worker.py batch_deployment=food_101_vit_small ++batch_deployment.workers_per_device=2
python3 src/serve/batch_server_multiple_worker.py batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=2
python3 src/serve/batch_server_multiple_worker.py batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=4


python3 src/serve/batch_server_parallel_decoding.py  batch_deployment=food_101_vit_small  ++batch_deployment.workers_per_device=4
python3 src/serve/batch_server_parallel_decoding.py  batch_deployment=food_101_vit_small  ++batch_deployment.workers_per_device=2
python3 src/serve/batch_server_parallel_decoding.py  batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=4
python3 src/serve/batch_server_parallel_decoding.py  batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=2


python3 src/serve/batch_server_half_precision.py batch_deployment=food_101_vit_small  ++batch_deployment.workers_per_device=4
python3 src/serve/batch_server_half_precision.py batch_deployment=food_101_vit_small  ++batch_deployment.workers_per_device=2
python3 src/serve/batch_server_half_precision.py  batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=4
python3 src/serve/batch_server_half_precision.py  batch_deployment=bird_200_mobilenetv3_large_optimized ++batch_deployment.workers_per_device=2




pip3 install setuptools==65.5.0
  113  pip3 install "antlr4-python3-runtime==4.9.2"
  114  pip3 install "hydra-core>=1.3.2,<1.3.3"
  115  clear
  116  pip3 install matplotlib
  117  pip3 install litserver
  118  pip3 install litserve
  119  clear
  120  python3 src/serve/batch_server.py batch_deployment=bird_200_mobilenetv3_large_optimized
  121  pip3 install rootutils