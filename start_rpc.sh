set -x
# kill & start rpc
ps aux | grep rpc_server | awk '{print $2}' | xargs kill
python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190 &
#CUDA_VISIBLE_DEVICES=0 python -m tvm.exec.rpc_server --tracker=0.0.0.0 --key=arnold --no-fork&
#CUDA_VISIBLE_DEVICES=1 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
#CUDA_VISIBLE_DEVICES=2 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
#CUDA_VISIBLE_DEVICES=3 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
CUDA_VISIBLE_DEVICES=4 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
CUDA_VISIBLE_DEVICES=5 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
CUDA_VISIBLE_DEVICES=6 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &
CUDA_VISIBLE_DEVICES=7 python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=arnold &

sleep 5
echo "=============query rpc workers============"
python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
